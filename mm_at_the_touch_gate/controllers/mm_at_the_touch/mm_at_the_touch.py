import time
from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from typing import override

import numpy as np
import pandas as pd
from pydantic import Field, model_validator
from scipy.linalg import expm

from hummingbot.core.data_type.common import MarketDict, PositionMode, TradeType
from hummingbot.core.event.event_forwarder import SourceInfoEventForwarder
from hummingbot.core.event.events import OrderBookEvent, OrderBookTradeEvent
from hummingbot.strategy_v2.controllers.controller_base import ControllerBase, ControllerConfigBase
from hummingbot.strategy_v2.executors.order_executor.data_types import ExecutionStrategy, OrderExecutorConfig
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, ExecutorAction, StopExecutorAction
from hummingbot.strategy_v2.models.executors_info import ExecutorInfo

# Order book snapshot schema. Five levels per side, fixed. Rows are stored as flat tuples in
# exactly this order so the DataFrame can be built in one pass with the schema already known.
OB_DEPTH = 5
OB_SNAPSHOT_COLUMNS: list[str] = (
    ["timestamp"]
    + [f"px_bid_{i}" for i in range(1, OB_DEPTH + 1)]
    + [f"px_ask_{i}" for i in range(1, OB_DEPTH + 1)]
    + [f"qty_bid_{i}" for i in range(1, OB_DEPTH + 1)]
    + [f"qty_ask_{i}" for i in range(1, OB_DEPTH + 1)]
)

# Public trade tape schema. "timestamp" is our local receipt clock, the same clock
# OB_SNAPSHOT_COLUMNS uses, so the two frames join on it directly. "ts_exchange" is the venue's
# own clock; the gap between them is feed latency. "side" is the taker's direction as +1 (bought,
# lifted the ask) or -1 (sold, hit the bid), so side * amount is signed volume.
TRADE_COLUMNS: list[str] = ["timestamp", "ts_exchange", "side", "price", "amount", "trade_id"]

# An empty DataFrame built from an empty list of tuples gets object dtype for every column, which
# makes merge_asof refuse the join ("both sides must have numeric dtype"). That case is not
# hypothetical: snapshots start accumulating before the first trade arrives. So the empty frames
# are cast explicitly, while the populated ones are left alone to avoid copying.
OB_SNAPSHOT_DTYPES = {column: "float64" for column in OB_SNAPSHOT_COLUMNS}
# level_id tags on the two quote executors, so the live order for each side can be found again.
LEVEL_BID = "touch_bid"
LEVEL_ASK = "touch_ask"

TRADE_DTYPES = {
    "timestamp": "float64", "ts_exchange": "float64", "side": "int64",
    "price": "float64", "amount": "float64", "trade_id": "object",
}


# ---------------------------------------------------------------------------------------------
# Research / modelling layer. Kept commented until each piece is implemented and tested.
#
# Pipeline:  ob_snapshots_df + trades_df
#              -> BookFeatures        (imbalance, midprice, regime)
#              -> SufficientStatistics (Delta tau_i, n_ij, M+_i, M-_i, over a per-regime window)
#              -> ImbalanceMarkovModel (lambda+-, Lambda, P, G, eps+-, mu)
#              -> AtTheTouchSolver     (h, and the two boolean posting maps)
#
# Everything down to ImbalanceMarkovModel is kept in step with car_mm: both controllers fit the
# same windowed imbalance model and differ only in the control problem solved on top of it. The
# copy is deliberate -- a Hummingbot controller is loaded as a standalone module, so the two files
# stay independently loadable rather than importing from one another.
#
# The solver needs exactly six arrays from the model: G, lam_plus, lam_minus, eps_plus,
# eps_minus, mu. Everything else the model computes is for validation and inspection.
#
# Imbalance regimes only -- no volatility state, so regime == state and n_states == n_regimes.
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RegimeDefinition:
    """How the continuous imbalance is bucketed into discrete regimes."""

    imbalance_bounds: list[float]   # n_regimes + 1 knots, fixed at [-1, -0.6, -0.2, 0.2, 0.6, 1]
    regime_labels: list[str]        # ordered strong sell -> strong buy

    def __post_init__(self):
        if len(self.imbalance_bounds) != len(self.regime_labels) + 1:
            raise ValueError(
                f"need one more bound than label: got {len(self.imbalance_bounds)} bounds "
                f"and {len(self.regime_labels)} labels"
            )

    @property
    def n_regimes(self) -> int:
        return len(self.regime_labels)

    @classmethod
    def default(cls) -> "RegimeDefinition":
        """Five equal-width regimes on [-1, 1], as in the order_imbalance notebook."""
        return cls(
            imbalance_bounds=[-1.0, -0.6, -0.2, 0.2, 0.6, 1.0],
            regime_labels=[
                "Strong sell pressure",
                "Mild sell pressure",
                "Neutral",
                "Mild buy pressure",
                "Strong buy pressure",
            ],
        )


class BookFeatures:
    """
    Raw controller frames -> the columns the estimators need.

    Stateless: with fixed equal-width bounds there is nothing to learn. If we later switch to
    quantile bounds (worth checking -- equal-width bins may leave the extreme regimes nearly
    empty on HYPE, which would make lambda+- and Lambda unstable exactly where it matters),
    this grows a fit() that learns the bounds from a training slice.
    """

    def __init__(self, regime_def: RegimeDefinition, imbalance_levels: int = OB_DEPTH):
        """
        imbalance_levels: how many book levels the imbalance sums over, 1..OB_DEPTH. Fewer levels
        means less averaging, so the imbalance swings wider and the outer regimes actually get
        visited -- which the estimators need, since a regime with no data blocks the fit. The cost
        is shorter sojourns: at a 1s sampling interval, visits lasting about one sample make
        Lambda and P unreliable. Watch 1/Lambda in the status when changing this.
        """
        if not 1 <= imbalance_levels <= OB_DEPTH:
            raise ValueError(f"imbalance_levels must be within 1..{OB_DEPTH}, got {imbalance_levels}")
        self.regime_def = regime_def
        self.imbalance_levels = imbalance_levels

    def transform_book(self, ob_snapshots_df: pd.DataFrame) -> pd.DataFrame:
        """
        Top-of-book features, one row per snapshot, sorted by timestamp.

        Returns columns ['timestamp', 'imbalance', 'midprice', 'spread', 'regime'].

        Imbalance is summed over the first imbalance_levels levels, not just the touch. This deviates from the
        notebook, which only had L1 bookTicker data -- depth-summed imbalance is smoother, which
        matters at a 1s sampling interval where a single L1 size change can flip the regime. It
        also means the notebook's fitted numbers are not directly comparable to ours.

        Sizes are summed with NaN skipped, so a book thinner than OB_DEPTH contributes only the
        levels it has. Imbalance is NaN when both sides sum to zero (0/0), which makes the regime
        NaN too; such rows are kept rather than dropped, so the caller decides how to handle them.

        'spread' is in dollars and is what Delta is measured from, when Delta is not passed in
        directly.
        """
        levels = range(1, self.imbalance_levels + 1)
        bid_qty = ob_snapshots_df[[f"qty_bid_{i}" for i in levels]].sum(axis=1)
        ask_qty = ob_snapshots_df[[f"qty_ask_{i}" for i in levels]].sum(axis=1)
        bid_px = ob_snapshots_df["px_bid_1"]
        ask_px = ob_snapshots_df["px_ask_1"]

        out = pd.DataFrame({
            "timestamp": ob_snapshots_df["timestamp"],
            "imbalance": (bid_qty - ask_qty) / (bid_qty + ask_qty),
            "midprice": (bid_px + ask_px) / 2,
            "spread": ask_px - bid_px,
        })
        out["regime"] = pd.cut(
            out["imbalance"],
            bins=self.regime_def.imbalance_bounds,
            labels=self.regime_def.regime_labels,
            include_lowest=True,
        )
        return out.sort_values("timestamp").reset_index(drop=True)


MO_COLUMNS: list[str] = ["timestamp", "side", "high", "low", "amount", "n_prints"]


def collapse_sweeps(trades: pd.DataFrame, sweep_window_s: float) -> pd.DataFrame:
    """
    Public prints -> market orders. The one definition of an MO every estimator here uses.

    The venue prints one trade per resting order a taker matches, so one MO that walks several
    levels, or hits several makers at one level, arrives as a burst of same-side prints. Counting
    each print as its own MO overstates the arrival rates exactly when the flow is most
    informative, and enters the same price move into eps once per print. So a new MO starts on a
    side change or a gap wider than sweep_window_s since the previous print; everything else
    joins the MO in progress.

    The gap is measured print to print, so a sweep whose prints keep arriving within the window
    stays one MO however long it runs. The clock is the local receipt one, which all the other
    frames share -- a sweep reported in one feed message lands with one timestamp.

    Returns one row per MO, oldest first:
        timestamp  first print, so the regime and the starting mid are the ones the MO met
        side       taker's direction, +1 bought / -1 sold
        high, low  price range across its prints -- how far it walked the book
        amount     total base volume
        n_prints   prints merged into it
    """
    if trades.empty:
        return pd.DataFrame(columns=MO_COLUMNS).astype(
            {"timestamp": "float64", "side": "int64", "high": "float64", "low": "float64",
             "amount": "float64", "n_prints": "int64"}
        )
    t = trades.sort_values("timestamp", kind="stable")
    new_mo = (t["timestamp"].diff() > sweep_window_s) | t["side"].ne(t["side"].shift())
    g = t.groupby(new_mo.cumsum().to_numpy())
    return pd.DataFrame({
        "timestamp": g["timestamp"].first(),
        "side": g["side"].first(),
        "high": g["price"].max(),
        "low": g["price"].min(),
        "amount": g["amount"].sum(),
        "n_prints": g.size(),
    }).reset_index(drop=True)


class SufficientStatistics:
    """
    Collapse the book/trade frames into the counts the rate estimators need, over a window that
    forgets old data. Pure counting, so this is the natural place to test against hand-worked
    examples.

    Why a window. Fitted on the whole history, every estimate is dominated by whatever the busiest
    stretch of the run was. After a pump and crash the post-crash market can trade for hours and
    still only move eps and mu a fraction of the way back, so the quotes stay wide and the drift
    stays pinned to a market that no longer exists.

    Why per regime. Every maximum-likelihood estimator is conditional on one regime: row i of G,
    lam+-_i and eps+-_i use only what happened during visits to regime i. So each regime can be
    windowed on its own, provided its window is a set of WHOLE visits -- then its MO counts, its
    exits and its exposure time all cover the same stretch and the count / exposure ratios stay
    valid rates.

    The window for regime i is chosen in two steps:
      1. Global: every visit from the one holding the window_mos-th most recent MO onward. The
         busy regimes all describe the same recent market.
      2. Evidence floor: if that leaves regime i short of min_mos buy MOs, min_mos sell MOs or
         min_sojourns completed visits, it alone reaches further back, one visit at a time, until
         the floor is met or the history runs out. A rarely visited regime therefore keeps its
         most recent evidence however old it is, rather than going undefined -- and the busy
         regimes are not dragged back with it.

    Visit durations are measured on the full snapshot sequence BEFORE any window is applied, so a
    visit always ends at the next snapshot in a different regime, whichever regime that is. A
    window only chooses which visits count; it never re-measures them, so extending one regime
    back cannot overstate its exposure time.
    """

    def __init__(self, regime_def: RegimeDefinition, window_mos: int | None = None,
                 min_mos: int = 1, min_sojourns: int = 1):
        """
        window_mos: the global window, in MOs (sweeps merged, see collapse_sweeps). None or 0
        uses the whole history for every regime, i.e. no forgetting.
        min_mos: the evidence floor, in MOs on EACH side, per regime. Never below 1: eps is a
        conditional mean, so it needs at least one MO on each side to exist.
        min_sojourns: the evidence floor, in completed visits, per regime. Never below 1: a regime
        with no completed visit has recorded no exit, so its Lambda would be 0 and G would make it
        absorbing.
        """
        self.regime_def = regime_def
        self.window_mos = window_mos or None
        self.min_mos = max(int(min_mos), 1)
        self.min_sojourns = max(int(min_sojourns), 1)

    def fit(self, book_features: pd.DataFrame, mos: pd.DataFrame) -> "SufficientStatistics":
        """Collapse to sojourns, choose each regime's window, then count.

        mos: market orders, one row per MO with sweeps already merged (collapse_sweeps); only
        'timestamp' and 'side' are read. Each MO is attributed to the visit its first print
        arrived in, so it carries that visit's regime by construction.

        Always sets
        -----------
        self.ready_     : bool        every regime meets the evidence floor
        self.blockers_  : list[str]   why not, in plain language
        self.coverage_  : DataFrame   per regime: completed visits, buy/sell MOs, the start of
                                      the oldest visit used, and whether the floor forced the
                                      window back past the global one

        Sets when ready
        ---------------
        self.delta_tau_ : ndarray (n_regimes,)              seconds spent in regime i
        self.n_ij_      : ndarray (n_regimes, n_regimes)    switches i -> j, zero diagonal
        self.M_plus_    : ndarray (n_regimes,)              buy MOs that arrived in regime i
        self.M_minus_   : ndarray (n_regimes,)              sell MOs that arrived in regime i
        self.n_sojourns_: ndarray (n_regimes,)              completed visits to regime i (== exits)
        self.sojourns_  : DataFrame                         the selected visits, for inspection
        self.mos_       : DataFrame                         the selected MOs, with their regime
        self.end_time_  : float                             last snapshot, where exposure stops

        The visit in progress is right-censored: we know it has lasted at least this long, not
        when it will end. Its time and its MOs still count -- the MLE of a rate divides events by
        ALL exposure time, censored or not -- but it records no exit. Dropping it outright, as the
        full-history version did, would bias Lambda of the current regime upward and discard the
        evidence most relevant to the regime the agent is quoting in right now.
        """
        labels = self.regime_def.regime_labels
        self.ready_, self.blockers_ = False, []
        self.coverage_ = pd.DataFrame(
            {"visits": 0, "buy_mos": 0, "sell_mos": 0, "window_start": np.nan,
             "extended": False, "met": False},
            index=pd.Index(labels, name="regime"),
        )

        # A NaN regime (both sides' sizes zero) is not a state the chain can visit. Dropping these
        # rows before collapsing matters: ne(shift()) treats NaN as its own value, so a single
        # NaN row sitting inside a genuine sojourn would split it into three.
        book = book_features.dropna(subset=["regime"]).sort_values("timestamp")
        if book.empty:
            self.blockers_.append("no order book snapshots with a defined imbalance yet")
            return self

        # One row per uninterrupted visit: keep the updates that entered a new regime. Measured
        # over the whole sample, before any windowing, so every duration is the true one.
        entered = book["regime"].ne(book["regime"].shift())
        sojourns = (
            book.loc[entered, ["timestamp", "regime"]]
            .rename(columns={"timestamp": "start"})
            .reset_index(drop=True)
        )
        self.end_time_ = float(book["timestamp"].iloc[-1])
        sojourns["next_regime"] = sojourns["regime"].shift(-1)
        sojourns["completed"] = sojourns["next_regime"].notna()
        sojourns["duration"] = sojourns["start"].shift(-1).fillna(self.end_time_) - sojourns["start"]
        n = len(sojourns)

        # Attribute each MO to the visit it arrived in. MOs before the first snapshot have no
        # visit, and MOs after the last one fall outside the exposure time, so both are dropped.
        mo = mos.loc[
            (mos["timestamp"] >= sojourns["start"].iloc[0])
            & (mos["timestamp"] <= self.end_time_),
            ["timestamp", "side"],
        ].sort_values("timestamp")
        mo = pd.merge_asof(
            mo, sojourns[["start"]].assign(sojourn=np.arange(n)),
            left_on="timestamp", right_on="start", direction="backward",
        ).drop(columns="start")
        mo["regime"] = pd.Categorical(
            sojourns["regime"].iloc[mo["sojourn"].to_numpy()].to_numpy(), categories=labels
        )

        sojourn_ids = mo["sojourn"].to_numpy()
        buys = np.bincount(sojourn_ids[mo["side"].to_numpy() > 0], minlength=n)
        sells = np.bincount(sojourn_ids[mo["side"].to_numpy() < 0], minlength=n)
        completed = sojourns["completed"].to_numpy()

        # The global window: the visit holding the window_mos-th most recent MO, and every visit
        # after it. Whole visits only, so the boundary never cuts one in two.
        if self.window_mos and len(mo) > self.window_mos:
            first_in_window = int(sojourn_ids[-self.window_mos])
        else:
            first_in_window = 0

        codes = sojourns["regime"].cat.codes.to_numpy()
        starts = sojourns["start"].to_numpy()
        selected = np.zeros(n, dtype=bool)
        for i, label in enumerate(labels):
            ids = np.flatnonzero(codes == i)[::-1]      # this regime's visits, newest first
            if not len(ids):
                self.blockers_.append(f"{label}: never visited yet")
                continue
            in_window = int((ids >= first_in_window).sum())
            # Cumulative evidence walking back one visit at a time. Each series only grows, so
            # the first visit at which all three floors hold is the shortest window that works.
            ok = (
                (np.cumsum(buys[ids]) >= self.min_mos)
                & (np.cumsum(sells[ids]) >= self.min_mos)
                & (np.cumsum(completed[ids]) >= self.min_sojourns)
            )
            met = bool(ok.any())
            needed = int(np.argmax(ok)) + 1 if met else len(ids)
            take = ids[:max(in_window, needed)]
            selected[take] = True
            self.coverage_.loc[label] = [
                int(completed[take].sum()), int(buys[take].sum()), int(sells[take].sum()),
                float(starts[take[-1]]), needed > in_window, met,
            ]
            if not met:
                self.blockers_.append(
                    f"{label}: {buys[take].sum()}/{self.min_mos} buy MOs,"
                    f" {sells[take].sum()}/{self.min_mos} sell MOs,"
                    f" {completed[take].sum()}/{self.min_sojourns} completed visits"
                    " across the whole history"
                )

        if self.blockers_:
            return self
        self.ready_ = True

        sel = sojourns[selected]
        self.sojourns_ = sel
        self.delta_tau_ = (
            sel.groupby("regime", observed=False)["duration"].sum()
            .reindex(labels).fillna(0.0).to_numpy(dtype=float)
        )

        # Exits come from completed visits only. Counting visit -> next visit makes the diagonal
        # zero, since a visit ends precisely when the regime changes.
        done = sel[sel["completed"]]
        self.n_ij_ = (
            pd.crosstab(done["regime"], done["next_regime"], dropna=False)
            .reindex(index=labels, columns=labels).fillna(0.0).to_numpy(dtype=float)
        )
        if np.diagonal(self.n_ij_).any():
            raise AssertionError("n_ij has a non-zero diagonal, so the sojourn collapse is wrong")
        self.n_sojourns_ = self.n_ij_.sum(axis=1)

        self.mos_ = mo.loc[selected[sojourn_ids]].reset_index(drop=True)
        self.M_plus_ = self._count_by_regime(self.mos_.loc[self.mos_["side"] > 0])
        self.M_minus_ = self._count_by_regime(self.mos_.loc[self.mos_["side"] < 0])
        return self

    def _count_by_regime(self, trades: pd.DataFrame) -> "np.ndarray":
        """Market orders per regime, in regime_labels order, zero where none arrived."""
        return (
            trades.groupby("regime", observed=False).size()
            .reindex(self.regime_def.regime_labels).fillna(0).to_numpy(dtype=float)
        )

    @property
    def oldest_start(self) -> float | None:
        """Start of the oldest visit any regime's window uses -- how far back the fit reaches."""
        starts = self.coverage_["window_start"].dropna().to_numpy(dtype=float)
        return float(starts.min()) if starts.size else None

    def as_dict(self) -> dict:
        return {
            "regime_labels": list(self.regime_def.regime_labels),
            "delta_tau": self.delta_tau_,
            "n_ij": self.n_ij_,
            "M_plus": self.M_plus_,
            "M_minus": self.M_minus_,
            "n_sojourns": self.n_sojourns_,
            "end_time": self.end_time_,
            "coverage": self.coverage_,
        }


class ImbalanceMarkovModel:
    """
    Continuous-time Markov chain of the imbalance regime, with MO arrivals as regime-dependent
    Poisson processes, plus the conditional price move per regime and MO side.
    """

    def __init__(self, regime_def: RegimeDefinition, tick_size: float,
                 delta_window: int | None = None):
        """
        delta_window: how many of the most recent snapshots Delta is measured over. None uses the
        whole sample. A short window matters on a tight-spread pair: over hours the median spread
        sits at the minimum tick, where the half-spread cannot cover a maker fee and the policy
        never posts. A one-minute window tracks the wider spells instead, so the model sees the
        edge while it is actually there.
        """
        self.regime_def = regime_def
        self.tick_size = tick_size
        self.delta_window = delta_window

    def fit(self, stats: "SufficientStatistics", book_features: pd.DataFrame) -> "ImbalanceMarkovModel":
        """
        Apply the maximum-likelihood rate estimators to windowed sufficient statistics and measure
        the conditional price moves over the SAME window, so mu = lam*eps is built from rates and
        moves that describe the same stretch of market.

        Sets -- the solver's inputs
        ---------------------------
        self.G_         : ndarray (n, n)   generator; off-diag n_ij/delta_tau_i, diag -Lambda_i
        self.lam_plus_  : ndarray (n,)     buy-MO arrival rate per regime, /s
        self.lam_minus_ : ndarray (n,)     sell-MO arrival rate per regime, /s
        self.eps_plus_  : ndarray (n,)     mean midprice move after a buy MO, dollars
        self.eps_minus_ : ndarray (n,)     same for sell MOs, sign-flipped to be positive
        self.mu_        : ndarray (n,)     lam_plus*eps_plus - lam_minus*eps_minus, dollars/s

        Sets -- diagnostics only
        ------------------------
        self.Lambda_    : ndarray (n,)     total exit rate per regime, /s
        self.P_         : ndarray (n, n)   embedded jump chain, zero diagonal, rows sum to 1
        self.delta_     : float            median quoted spread, dollars (Delta for the solver)
        self.jump_dist_ : pd.DataFrame     empirical price-change distribution per regime
        self.eps_plus_se_, self.eps_minus_se_ : ndarray (n,)   standard errors of eps+-
        self.mu_se_     : ndarray (n,)     approximate standard error of mu
        self.stats_     : SufficientStatistics

        No dt is needed anywhere: G comes straight from the sojourn counts. (The logm(A)/dt
        route is a separate cross-check, not part of this path.)
        """
        if not stats.ready_:
            raise ValueError(
                "the evidence floor is not met, so there is nothing to fit: "
                + "; ".join(stats.blockers_)
            )
        self.stats_ = stats

        # Maximum likelihood: every estimator is a count divided by the exposure time of the state
        # in which those events could happen. The evidence floor guarantees every regime at least one
        # completed visit, so it has both exposure time and an exit -- both denominators below
        # are safe.
        exits = stats.n_ij_.sum(axis=1)
        self.lam_plus_ = stats.M_plus_ / stats.delta_tau_
        self.lam_minus_ = stats.M_minus_ / stats.delta_tau_
        self.Lambda_ = exits / stats.delta_tau_
        self.P_ = stats.n_ij_ / exits[:, None]

        self.G_ = stats.n_ij_ / stats.delta_tau_[:, None]
        np.fill_diagonal(self.G_, -self.Lambda_)

        (self.eps_plus_, self.eps_minus_, self.eps_plus_se_, self.eps_minus_se_,
         self.jump_dist_) = self._fit_price_moves(book_features, stats.mos_)
        self.mu_ = self.lam_plus_ * self.eps_plus_ - self.lam_minus_ * self.eps_minus_

        # Delta method on mu = lam+ eps+ - lam- eps-, with Poisson se(lam) = sqrt(M)/delta_tau.
        # Optimistic: it treats every MO as independent, but MOs less than horizon_s apart share
        # most of their price move. (Sweeps are already merged, so one MO's prints no longer
        # count several times.) Read it as a lower bound -- what it is good for is seeing whether
        # mu is distinguishable from zero at all.
        se_lam_plus = np.sqrt(stats.M_plus_) / stats.delta_tau_
        se_lam_minus = np.sqrt(stats.M_minus_) / stats.delta_tau_
        self.mu_se_ = np.sqrt(
            (self.eps_plus_ * se_lam_plus) ** 2 + (self.lam_plus_ * self.eps_plus_se_) ** 2
            + (self.eps_minus_ * se_lam_minus) ** 2 + (self.lam_minus_ * self.eps_minus_se_) ** 2
        )
        spread = book_features["spread"].to_numpy(dtype=float)
        if self.delta_window:
            spread = spread[-self.delta_window:]
        self.delta_ = float(np.nanmedian(spread))

        return self

    def _fit_price_moves(self, book_features: pd.DataFrame, mos: pd.DataFrame,
                         horizon_s: float = 1.0):
        """
        Midprice change over horizon_s after each MO, conditional on (MO side, regime).

        mos is the windowed MO sample from SufficientStatistics, each already carrying the regime
        of the visit it arrived in -- so eps covers exactly the MOs the arrival rates count, and
        the two halves of mu cannot drift apart in time.

        Bucketed in half-ticks rather than bps: the midprice moves in half-ticks, so integer
        buckets fall out naturally and the exactly-unchanged case is just d == 0. Returns the
        conditional means (what the solver wants), their standard errors, and the full empirical
        distribution (for us).

        Sell MOs push the price down, so eps_minus is negated to be a positive magnitude.

        Everything price-dimensioned in this model is in DOLLARS, not bps. The DPE is
        homogeneous of degree 1 in the price unit -- scaling Delta, eps, phi, varphi and the
        fees by a common factor scales h by that factor and leaves the sign of c+- unchanged,
        so the policy is unit-invariant. Dollars are preferred over bps because bps needs a
        reference price, and that reference drifts over a long run. Values land on exact
        multiples of tick/2, since that is how the midprice moves.

        Unlike the rate estimators this is a conditional mean, not a count over an exposure
        time, so the right-censoring rule does not apply and every MO is usable. MOs within
        horizon_s of the end of the sample are dropped though: a backward merge for the later
        midprice would silently return a stale one and understate the move. That matters more
        here than in the notebook, since refitting every few seconds means the newest MOs are
        always the ones at risk.
        """
        labels = self.regime_def.regime_labels
        mid = book_features.loc[:, ["timestamp", "midprice"]].sort_values("timestamp")
        last_ts = float(mid["timestamp"].iloc[-1])

        mo = mos.sort_values("timestamp")
        mo = mo.loc[mo["timestamp"] + horizon_s <= last_ts, ["timestamp", "side", "regime"]].copy()

        mo = pd.merge_asof(mo, mid, on="timestamp", direction="backward")
        mo = mo.rename(columns={"midprice": "mid_0"})
        mo["t1"] = mo["timestamp"] + horizon_s
        mo = pd.merge_asof(
            mo.sort_values("t1"), mid.rename(columns={"timestamp": "t1"}),
            on="t1", direction="backward",
        ).rename(columns={"midprice": "mid_1"})
        mo["d"] = mo["mid_1"] - mo["mid_0"]

        def by_regime(rows: pd.DataFrame, how: str) -> "np.ndarray":
            return (
                rows.groupby("regime", observed=False)["d"].agg(how)
                .reindex(labels).to_numpy(dtype=float)
            )

        buys, sells = mo.loc[mo["side"] > 0], mo.loc[mo["side"] < 0]
        eps_plus = by_regime(buys, "mean")
        # Sell MOs push the price down, so flip the sign to get a positive adverse magnitude.
        # Subtract from zero rather than negate, so an unchanged mean gives 0.0 and not -0.0.
        eps_minus = 0.0 - by_regime(sells, "mean")
        # The sign flip does not touch the standard error. nan where a cell has a single MO.
        eps_plus_se = by_regime(buys, "sem")
        eps_minus_se = by_regime(sells, "sem")

        # A regime with no MO on one side has no conditional mean, and a nan here cannot simply
        # be zeroed: mu = lam_plus*eps_plus - lam_minus*eps_minus would keep the nan even where
        # the rate is zero, because 0 * nan is nan. Treating a missing eps as 0 is worse than
        # raising -- the solver would read "no adverse selection" and happily post there. So this
        # is the same precondition as delta_tau, one level stricter: every regime needs MOs on
        # both sides. The evidence floor upstream normally guarantees it; this still trips when
        # a regime's only MOs on one side arrived within horizon_s of the end of the sample.
        missing = (
            [f"buy MOs in {label}" for label, e in zip(labels, eps_plus) if not np.isfinite(e)]
            + [f"sell MOs in {label}" for label, e in zip(labels, eps_minus) if not np.isfinite(e)]
        )
        if missing:
            raise ValueError(
                f"no {', no '.join(missing)}: every regime needs market orders on both sides "
                "before the conditional price moves are defined. Keep gathering data."
            )

        # Empirical distribution in half-ticks, kept as counts so the per-cell sample size stays
        # visible -- eps is only as trustworthy as the number of MOs behind it.
        half_tick = self.tick_size / 2.0
        mo["d_half_ticks"] = np.round(mo["d"] / half_tick).astype("Int64")
        mo["mo_side"] = np.where(mo["side"] > 0, "Buy", "Sell")
        jump_dist = pd.crosstab([mo["mo_side"], mo["regime"]], mo["d_half_ticks"], dropna=False)

        return eps_plus, eps_minus, eps_plus_se, eps_minus_se, jump_dist

    def transition_matrix(self, horizon_s: float) -> "np.ndarray":
        """expm(G * horizon_s): P(regime j at t+horizon | regime i at t). No matrix_power."""
        return expm(self.G_ * horizon_s)

    def mo_rates(self, regime: int) -> dict:
        """{'buy': lam_plus_[regime], 'sell': lam_minus_[regime]}, per second."""
        return {"buy": float(self.lam_plus_[regime]), "sell": float(self.lam_minus_[regime])}

    PARAM_KEYS = ("G", "lam_plus", "lam_minus", "eps_plus", "eps_minus", "mu", "Lambda", "P")

    def get_params(self) -> dict:
        """
        Everything the solver needs, plus enough context to interpret it. Lists rather than
        ndarrays so the dict is JSON-serialisable.

        The data-dependent diagnostics (stats_, jump_dist_) are deliberately left out: they
        describe the sample, not the model, and they are what you would refit from.
        """
        params = {
            "imbalance_bounds": list(self.regime_def.imbalance_bounds),
            "regime_labels": list(self.regime_def.regime_labels),
            "tick_size": self.tick_size,
            "delta_window": self.delta_window,
            "delta": self.delta_,
        }
        for key in self.PARAM_KEYS:
            params[key] = np.asarray(getattr(self, f"{key}_")).tolist()
        return params

    @classmethod
    def from_params(cls, params: dict) -> "ImbalanceMarkovModel":
        """Rebuild a fitted model from get_params() output, without touching any data."""
        model = cls(
            regime_def=RegimeDefinition(
                imbalance_bounds=list(params["imbalance_bounds"]),
                regime_labels=list(params["regime_labels"]),
            ),
            tick_size=params["tick_size"],
            delta_window=params.get("delta_window"),
        )
        model.delta_ = float(params["delta"])
        for key in cls.PARAM_KEYS:
            setattr(model, f"{key}_", np.asarray(params[key], dtype=float))
        return model


class AtTheTouchSolver:
    """
    Optimal at-the-touch postings under the fitted regime model.

    The agent may only quote at the best bid/ask under a constant spread Delta, so the control
    on each side is binary rather than a depth, and a matching MO fills her with certainty.
    Posting gains, per side:

        c+(t,z,q) = Delta/2 - eps_plus[z]  - fee + h[t,z,q-1] - h[t,z,q]
        c-(t,z,q) = Delta/2 - eps_minus[z] - fee + h[t,z,q+1] - h[t,z,q]

    All price-dimensioned quantities are in dollars, matching the model.

    and she posts exactly when the gain is positive, subject to the inventory band:

        ell+ = 1{c+ > 0} 1{q > q_min}        ell- = 1{c- > 0} 1{q < q_max}

    ell+ is the ASK side: N+ counts her sell fills, driven by lam_plus (buy MOs lift her ask).
    ell- is the BID side. Easy to invert by accident.
    """

    def __init__(self, model: "ImbalanceMarkovModel", risk_params: dict):
        """
        Delta is read off model.delta_ rather than passed here -- it is measured, not chosen.

        risk_params
        -----------
        phi        running inventory penalty, dollars/s
        varphi     terminal walking-the-book penalty, dollars
        q_min      inventory floor, units
        q_max      inventory ceiling, units
        T          horizon, seconds
        dt         backward Euler step, seconds. Explicit Euler needs
                   dt * max(Lambda_i + lam_plus_i + lam_minus_i) < 1, which solve() asserts
                   rather than assuming the notebook's 0.01 carries over -- HYPE's regimes may
                   switch far faster than BTC's, which inflates Lambda.
        fee_maker  maker fee per fill, dollars (comes straight off Delta/2 at the touch).
                   Fees are quoted as a fraction of notional, so this is the one quantity
                   whose conversion needs a reference price: fee_dollars = f * S.
        fee_taker  taker fee, dollars; paid only on the terminal liquidation
        drift_scale weight on the fitted mu, default 1.0. The model's mu_ is left untouched;
                   scaling here keeps the estimate and the decision to trust it separate.
        """
        self.model = model
        self.risk_params = dict(risk_params)

        self.phi = float(risk_params["phi"])
        self.varphi = float(risk_params["varphi"])
        self.q_min = int(risk_params["q_min"])
        self.q_max = int(risk_params["q_max"])
        self.T = float(risk_params["T"])
        self.dt = float(risk_params["dt"])
        self.fee_maker = float(risk_params.get("fee_maker", 0.0))
        self.fee_taker = float(risk_params.get("fee_taker", 0.0))
        self.drift_scale = float(risk_params.get("drift_scale", 1.0))

        if self.q_min > 0 or self.q_max < 0:
            raise ValueError(f"inventory band [{self.q_min}, {self.q_max}] must contain 0")

        self.q_ = np.arange(self.q_min, self.q_max + 1)
        self.t_ = np.linspace(0.0, self.T, round(self.T / self.dt) + 1)

    def solve(self) -> "AtTheTouchSolver":
        """
        Backward explicit Euler for h on the (t, regime, q) grid, from h(T,z,q) = -g(q):

            0 = dh/dt + mu[z]*q - phi*q^2
                + lam_plus[z]  * max(c+, 0) * 1{q > q_min}
                + lam_minus[z] * max(c-, 0) * 1{q < q_max}
                + sum_k G[z,k] * (h[t,k,q] - h[t,z,q])

        The only non-linearity is max(.,0), so the rhs is piecewise linear and well behaved.

        Sets
        ----
        self.h_         : ndarray (n_t, n_regimes, n_q)   value function
        self.ell_plus_  : ndarray (n_t, n_regimes, n_q)   bool, post the ask
        self.ell_minus_ : ndarray (n_t, n_regimes, n_q)   bool, post the bid
        """
        m = self.model
        half_spread = m.delta_ / 2.0
        q, t = self.q_, self.t_

        rate = float(np.max(m.Lambda_ + m.lam_plus_ + m.lam_minus_))
        if self.dt * rate >= 1.0:
            raise ValueError(
                f"dt={self.dt} is too large for these rates: dt * max(Lambda + lam+ + lam-) = "
                f"{self.dt * rate:.3f} must be < 1 for explicit Euler to be stable. Use "
                f"dt < {1.0 / rate:.5f}."
            )

        # Terminal liquidation cost: the half-spread plus the taker fee whichever way the position
        # is unwound, plus the walking-the-book penalty.
        g = np.abs(q) * (half_spread + self.fee_taker) + self.varphi * q**2

        h = np.empty((len(t), m.regime_def.n_regimes, len(q)))
        h[-1] = -g
        edge = half_spread - self.fee_maker
        base = (self.drift_scale * m.mu_)[:, None] * q - self.phi * q**2

        for n in range(len(t) - 2, -1, -1):
            hn = h[n + 1]
            # c+ only exists where the agent may still sell (q > q_min); c- where she may buy.
            c_plus = edge - m.eps_plus_[:, None] + hn[:, :-1] - hn[:, 1:]
            c_minus = edge - m.eps_minus_[:, None] + hn[:, 1:] - hn[:, :-1]

            dh = base + m.G_ @ hn
            dh[:, 1:] += m.lam_plus_[:, None] * np.maximum(c_plus, 0.0)
            dh[:, :-1] += m.lam_minus_[:, None] * np.maximum(c_minus, 0.0)
            h[n] = hn + self.dt * dh

        self.h_ = h
        c_plus, c_minus = self._posting_gains(h)
        # Strictly positive: do not post when indifferent.
        self.ell_plus_ = c_plus > 0.0
        self.ell_minus_ = c_minus > 0.0
        return self

    def _posting_gains(self, h: "np.ndarray"):
        """c+ and c- on the full (t, z, q) grid, zero at the boundaries where posting is barred."""
        m = self.model
        edge = m.delta_ / 2.0 - self.fee_maker
        c_plus, c_minus = np.zeros_like(h), np.zeros_like(h)
        c_plus[:, :, 1:] = edge - m.eps_plus_[:, None] + h[:, :, :-1] - h[:, :, 1:]
        c_minus[:, :, :-1] = edge - m.eps_minus_[:, None] + h[:, :, 1:] - h[:, :, :-1]
        return c_plus, c_minus

    def should_post(self, time_remaining: float, regime: int, inventory: int) -> tuple[bool, bool]:
        """
        The live decision: index the solved policy, no optimisation at quote time.
        Returns (post_bid, post_ask) -- note the order is bid first, solver-internal is ell+/ask.

        time_remaining is seconds left in the horizon, so it maps to grid time T - time_remaining
        and is clamped to the grid. inventory must sit inside the band the policy was solved on.
        """
        if not self.q_min <= inventory <= self.q_max:
            raise ValueError(
                f"inventory {inventory} is outside the solved band "
                f"[{self.q_min}, {self.q_max}]; the policy says nothing about it"
            )
        n = round((self.T - float(time_remaining)) / self.dt)
        n = min(max(n, 0), len(self.t_) - 1)
        iq = int(inventory) - self.q_min
        return bool(self.ell_minus_[n, regime, iq]), bool(self.ell_plus_[n, regime, iq])

    def get_params(self) -> dict:
        """
        Posting maps plus grid metadata, so the controller can load a solved policy.

        The maps are ndarrays, not lists: at the notebook's T=600, dt=0.01 they are
        60001 x n_regimes x n_q booleans each, which is fine in memory but far too large for
        JSON. Persist with np.savez_compressed rather than json.dump.
        """
        return {
            "ell_plus": self.ell_plus_,
            "ell_minus": self.ell_minus_,
            "q": self.q_,
            "t": self.t_,
            "regime_labels": list(self.model.regime_def.regime_labels),
            "delta": self.model.delta_,
            "risk_params": dict(self.risk_params),
        }


class MMAtTheTouchConfig(ControllerConfigBase):
    """
    At the touch market making strategy.

    Notes:
        - add enhanced timing parameters
        - add price distance tolerance
        - add refresh tolerance
    """
    controller_type: str = "generic"
    controller_name: str = "mm_at_the_touch"

    # Market settings
    connector_name: str = Field(
        default="hyperliquid_perpetual",
        json_schema_extra={
            "prompt_on_new": True,
            "prompt": "Enter the connector name (e.g., binance):",
        }
    )
    trading_pair: str = Field(
        default="HYPE-USD",
        json_schema_extra={
            "prompt_on_new": True,
            "prompt": "Enter the trading pair (e.g., BTC-USDT):",
        }
    )
    leverage: int = Field(default=1, json_schema_extra={"is_updatable": True})
    position_mode: PositionMode = Field(default=PositionMode.ONEWAY)

    # Spread and Amount configuration
    total_amount_quote: Decimal = Field(
        default=Decimal(0),
        json_schema_extra={"prompt_on_new": False}
    )
    order_amount: Decimal = Field(
        default=Decimal("0.5"),
        json_schema_extra={
            "prompt_on_new": True, "is_updatable": True,
            "prompt": "Order amount in base asset, one unit of q (e.g. 0.5 for ~$45 of HYPE):",
        }
    )
    # The holding that counts as q = 0, in base units. Inventory is measured as the deviation from
    # it: with a baseline of 20 HYPE, holding 19.5 HYPE on a 0.5 unit is q = -1. That is how a spot
    # market, which cannot go short, gets a two-sided band -- "short" just means holding less than
    # the baseline. On a perpetual it can stay 0, which is plain netted-position inventory exactly
    # as before; a non-zero value there means "quote around being long (or short) this much".
    # Not updatable: the anchored holding and the inventory read from it assume it is fixed.
    baseline_holding: Decimal = Field(
        default=Decimal(0),
        json_schema_extra={
            "prompt_on_new": False,
            "prompt": "Holding that counts as zero inventory, in base units (required on spot):",
        }
    )
    # Quotes are always repriced to the current touch, with no refresh tolerance: at the touch the
    # only price worth holding is the best one, so a stale quote has no value. A live quote already
    # sitting at the touch is left alone, because requoting the same level only costs queue place.
    enable_bid: bool = Field(
        default=True,
        json_schema_extra={
            "prompt_on_new": True, "is_updatable": True,
            "prompt": "Quote the bid side? (True/False):",
        }
    )
    enable_ask: bool = Field(
        default=True,
        json_schema_extra={
            "prompt_on_new": True, "is_updatable": True,
            "prompt": "Quote the ask side? (True/False):",
        }
    )
    # --- model / policy ---
    imbalance_levels: int = Field(
        default=2,
        json_schema_extra={
            "prompt_on_new": False,
            "prompt": "Book levels the imbalance sums over, 1-5 (fewer = wider swings, shorter sojourns):",
        }
    )
    # Kept equal to model_refit_interval on purpose: each fit then measures Delta over exactly the
    # period it governs. A window much shorter than the refit interval would hold a near-instant
    # spread reading for many seconds after it stopped being true.
    delta_window_snapshots: int = Field(
        default=15,
        json_schema_extra={
            "prompt_on_new": False, "is_updatable": True,
            "prompt": "Snapshots to measure Delta (the spread) over, most recent first:",
        }
    )
    # Defines what counts as ONE market order everywhere: the imbalance model's arrival rates and
    # eps, the estimation windows and the drift ramp all count MOs with same-side prints this close
    # together merged (see collapse_sweeps). A few tens of milliseconds catches a book walk arriving
    # as separate prints without merging distinct MOs.
    sweep_window_seconds: float = Field(
        default=0.05,
        json_schema_extra={
            "prompt_on_new": False, "is_updatable": True,
            "prompt": "Same-side prints within this many seconds are one swept market order:",
        }
    )
    # --- estimation window ---
    # The imbalance model forgets old data. Each regime is fitted on the visits since the
    # window_mos-th most recent market order; a regime short of the evidence floor inside that
    # window reaches further back on its own, one whole visit at a time, until it has enough. See
    # SufficientStatistics. The same floor is also the start-up gate: nothing is fitted, and so
    # nothing is quoted, until every regime meets it. Both count market orders with sweeps merged,
    # not raw prints.
    window_mos: int = Field(
        default=2000,
        json_schema_extra={
            "prompt_on_new": False, "is_updatable": True,
            "prompt": "Fit the imbalance model on the most recent this-many market orders (0 = whole history):",
        }
    )
    min_mos_per_regime: int = Field(
        default=100,
        json_schema_extra={
            "prompt_on_new": False, "is_updatable": True,
            "prompt": "Evidence floor: buy AND sell market orders each regime needs before it stops reaching back:",
        }
    )
    min_sojourns_per_regime: int = Field(
        default=10,
        json_schema_extra={
            "prompt_on_new": False, "is_updatable": True,
            "prompt": "Evidence floor: completed visits each regime needs (sets the exit-rate precision):",
        }
    )
    model_refit_interval: float = Field(
        default=15.0,
        json_schema_extra={
            "prompt_on_new": False, "is_updatable": True,
            "prompt": "Seconds between refits once the model is running:",
        }
    )
    fee_maker_pct: Decimal = Field(
        default=Decimal(0),
        json_schema_extra={
            "prompt_on_new": False, "is_updatable": True,
            "prompt": "Maker fee as a fraction of notional (0.00015 = 0.015%, 0 = free):",
        }
    )
    fee_taker_pct: Decimal = Field(
        default=Decimal("0.00028"),
        json_schema_extra={
            "prompt_on_new": False, "is_updatable": True,
            "prompt": "Taker fee as a fraction of notional (0.00028 = 0.028%):",
        }
    )
    phi: float = Field(
        default=1e-5,
        json_schema_extra={
            "prompt_on_new": False, "is_updatable": True,
            "prompt": "Running inventory penalty, dollars per second per unit^2:",
        }
    )
    varphi: float = Field(
        default=1e-4,
        json_schema_extra={
            "prompt_on_new": False, "is_updatable": True,
            "prompt": "Terminal inventory penalty, dollars per unit^2:",
        }
    )
    # The drift term mu(z)*q dominates the value function at long horizons, and mu is estimated
    # from the sample's realised price move -- mostly noise on a thin sample. So it is ramped in
    # linearly over the first drift_scale_trades market orders (sweeps merged) rather than trusted
    # from the first fit. Counted in orders, not seconds, so a quiet start does not unlock the
    # drift on time alone before the evidence behind it exists -- and not in raw prints, so one
    # violent sweep does not count as dozens of observations.
    drift_scale_start: float = Field(
        default=0.0,
        json_schema_extra={
            "prompt_on_new": False, "is_updatable": True,
            "prompt": "Drift (mu) weight at the start of data collection:",
        }
    )
    drift_scale_end: float = Field(
        default=1.0,
        json_schema_extra={
            "prompt_on_new": False, "is_updatable": True,
            "prompt": "Drift (mu) weight once fully ramped in:",
        }
    )
    drift_scale_trades: int = Field(
        default=1000,
        json_schema_extra={
            "prompt_on_new": False, "is_updatable": True,
            "prompt": "Market orders (sweeps merged) over which the drift weight ramps from start to end:",
        }
    )

    q_min: int = Field(default=-10, json_schema_extra={"prompt_on_new": False})
    q_max: int = Field(default=10, json_schema_extra={"prompt_on_new": False})
    solver_horizon: float = Field(
        default=600.0,
        json_schema_extra={
            "prompt_on_new": False,
            "prompt": "Solver horizon T in seconds:",
        }
    )
    solver_dt: float = Field(
        default=0.1,
        json_schema_extra={
            "prompt_on_new": False,
            "prompt": "Solver backward-Euler step dt in seconds:",
        }
    )

    # After a fill on one side, that side stands down for this long. The two sides cool off
    # independently, so a bid fill does not stop the ask from quoting.
    fill_cooldown_seconds: float = Field(
        default=5.0,
        json_schema_extra={
            "prompt_on_new": False, "is_updatable": True,
            "prompt": "Seconds a side must wait after one of its orders fills:",
        }
    )

    # The status is only ever on screen, and front ends such as Condor do not show it. So the
    # same facts are written to the log every this-many seconds, as key=value pairs meant for
    # grepping and parsing rather than reading.
    status_log_interval: float = Field(
        default=60.0,
        json_schema_extra={
            "prompt_on_new": False, "is_updatable": True,
            "prompt": "Seconds between status snapshots written to the log (0 = off):",
        }
    )

    @property
    def is_perpetual(self) -> bool:
        """Same test the executors use, so the controller and the orders agree on the market type."""
        return "perpetual" in self.connector_name.lower()

    @model_validator(mode="after")
    def _baseline_covers_band_on_spot(self):
        """
        On spot, refuse a baseline too small to sell down to q_min. The bottom of the band means
        holding baseline - |q_min| units; below zero that is a short, which spot cannot do. One extra
        unit is required as a buffer, because fills can overshoot the band (partial fills, cancels
        in flight) and on spot an overshoot shows up as rejected sell orders.
        """
        if self.is_perpetual or self.q_min >= 0:
            return self
        need = (abs(self.q_min) + 1) * self.order_amount
        if self.baseline_holding < need:
            raise ValueError(
                f"baseline_holding ({self.baseline_holding}) must be at least {need} on a spot "
                f"market: q_min = {self.q_min} means selling {abs(self.q_min)} units of "
                f"{self.order_amount} below the baseline, plus one unit of buffer, and spot cannot "
                "go short."
            )
        return self

    def update_markets(self, markets: MarketDict) -> MarketDict:
        return markets.add_or_update(self.connector_name, self.trading_pair)  # type: ignore


class MMAtTheTouch(ControllerBase):
    """
    Market Making At the Touch Controller.
    """

    def __init__(self, config: MMAtTheTouchConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config

        # Public trade tape. The forwarder adapts the (event_tag, caller, event) listener signature
        # down to our handler. Subscription happens on the first update_processed_data call, not
        # here: the connector's order book does not exist yet at construction time.
        self.order_book_trade_event = SourceInfoEventForwarder(self._process_public_trade)
        self._subscribed_to_trades: bool = False
        self._last_trade: OrderBookTradeEvent | None = None
        self._trade_count: int = 0
        # Market orders seen, with sweeps merged by the same rule as collapse_sweeps, counted live
        # so the drift ramp never runs back. The last print's side and local time are what the
        # next print is compared against.
        self._mo_count: int = 0
        self._last_print: tuple[int, float] | None = None

        # Running memory of order book snapshots, one tuple per capture in OB_SNAPSHOT_COLUMNS
        # order. Unbounded on purpose: this is research data for a run of at most ~48h.
        self._ob_snapshots: deque[tuple[float, ...]] = deque()

        # Running memory of the public trade tape, one tuple per trade in TRADE_COLUMNS order.
        self._trades: deque[tuple] = deque()

        # Model / policy state. The model is first fitted once every regime meets the evidence
        # floor -- enough MOs on both sides and enough completed visits, see SufficientStatistics.
        # _stats is the latest evaluation of that floor, kept for the status whether it passed.
        self.regime_def = RegimeDefinition.default()
        self.book_features = BookFeatures(self.regime_def, config.imbalance_levels)
        self._last_fit_attempt: float = 0.0
        self._last_fit_ts: float | None = None
        self._fit_seconds: float | None = None
        self._fit_error: str | None = None
        self._stats: SufficientStatistics | None = None
        self._tick_size: float | None = None
        self.model: ImbalanceMarkovModel | None = None
        self.solver: AtTheTouchSolver | None = None

        # Per-side fill cooloff. Fills are detected from the change in net position: our own
        # orders are the only thing that moves it, so a rise means the bid filled and a fall means
        # the ask did.
        self._last_position: Decimal | None = None
        self._last_bid_fill_ts: float | None = None
        self._last_ask_fill_ts: float | None = None

        # What we actually hold minus what our own fills say, in base units. On a perpetual it is
        # zero: the netted position IS the holding, exactly as before. On spot it is anchored once,
        # on the first tick, from the exchange's total base balance -- see _anchor_holding. None
        # until then, and nothing is quoted while it is None, since inventory is unknown.
        self._holding_offset: Decimal | None = Decimal(0) if config.is_perpetual else None

        self._last_status_log: float = 0.0

        # Own activity, so dormancy can be read straight off the log instead of inferred from the
        # policy. Per side: fill events, filled notional, and seconds with a live quote resting.
        # The interval counters reset at every status snapshot; the totals run for the whole
        # session, so a reader polling less often than the snapshot can still difference them.
        self._interval = self._empty_activity()
        self._totals = self._empty_activity()
        self._last_activity_ts: float | None = None

    @staticmethod
    def _padded_levels(frame: pd.DataFrame, column: str) -> list[float]:
        """
        First OB_DEPTH values of a snapshot column as floats, NaN-padded if the book is thinner.
        Padding rather than skipping keeps one row per capture, so the time grid stays regular.
        """
        values = frame[column].to_numpy(dtype=float)[:OB_DEPTH].tolist()
        return values + [float("nan")] * (OB_DEPTH - len(values))

    def _record_ob_snapshot(self, bids_df: pd.DataFrame, asks_df: pd.DataFrame):
        """
        Flatten one snapshot into a single row and append it to the running buffer. The snapshot
        frames arrive already sorted from the touch outward, so level 1 is row 0 on each side.
        """
        self._ob_snapshots.append((
            self.market_data_provider.time(),
            *self._padded_levels(bids_df, "price"),
            *self._padded_levels(asks_df, "price"),
            *self._padded_levels(bids_df, "amount"),
            *self._padded_levels(asks_df, "amount"),
        ))

    @property
    def ob_snapshots_df(self) -> pd.DataFrame:
        """
        Every snapshot captured so far, oldest first. Built on demand rather than appended to:
        appending to a DataFrame reallocates it each time, which is O(n^2) over a long run.
        """
        rows = list(self._ob_snapshots)
        frame = pd.DataFrame(rows, columns=OB_SNAPSHOT_COLUMNS)
        return frame if rows else frame.astype(OB_SNAPSHOT_DTYPES)

    def _subscribe_to_trades(self):
        """
        Attach our listener to the connector's order book trade stream. Returns True once attached.
        """
        try:
            order_book = self.market_data_provider.get_order_book(
                self.config.connector_name, self.config.trading_pair
            )
        except ValueError:
            return False  # book not tracked yet; retry next tick
        order_book.add_listener(OrderBookEvent.TradeEvent, self.order_book_trade_event)
        self._subscribed_to_trades = True
        self.logger().info(
            f"Subscribed to public trades for {self.config.connector_name} {self.config.trading_pair}"
        )
        return True

    def _process_public_trade(self, _event_tag: int, _market, event: OrderBookTradeEvent):
        """
        Called on every public trade that crosses the book. Runs on the event loop, so keep it
        cheap: this only stamps the local clock, counts market orders and appends one tuple.
        """
        now = self.market_data_provider.time()
        side = 1 if event.type == TradeType.BUY else -1
        # The same boundary collapse_sweeps draws: a side change or a gap past the window starts
        # a new MO, anything else is another print of the one in progress.
        if (self._last_print is None or side != self._last_print[0]
                or now - self._last_print[1] > self.config.sweep_window_seconds):
            self._mo_count += 1
        self._last_print = (side, now)
        self._last_trade = event
        self._trade_count += 1
        self._trades.append((
            now,
            float(event.timestamp),
            side,
            float(event.price),
            float(event.amount),
            event.trade_id,
        ))

    @property
    def trades_df(self) -> pd.DataFrame:
        """
        Every public trade seen so far, oldest first. Built on demand, same as ob_snapshots_df.
        """
        rows = list(self._trades)
        frame = pd.DataFrame(rows, columns=TRADE_COLUMNS)
        return frame if rows else frame.astype(TRADE_DTYPES)

    @override
    async def update_processed_data(self):
        if not self._subscribed_to_trades:
            self._subscribe_to_trades()

        # orderbook snapshot
        bids_df, asks_df = self.market_data_provider.get_order_book_snapshot(
            self.config.connector_name, self.config.trading_pair
        )
        if not bids_df.empty and not asks_df.empty:
            self._record_ob_snapshot(bids_df, asks_df)

        self.processed_data = {
            "last_trade": self._last_trade,
            "trade_count": self._trade_count,
            "ob_snapshot_count": len(self._ob_snapshots),
            "trade_row_count": len(self._trades),
        }

        self._anchor_holding()
        self._track_fills()
        self._track_activity()
        self._maybe_fit_model()
        self._maybe_log_status()

    # Model fitting -------------------------------------------------------------------------

    @property
    def tick_size(self) -> float | None:
        """min_price_increment for the traded pair, cached after the first successful read."""
        if self._tick_size is None:
            try:
                rules = self.market_data_provider.get_trading_rules(
                    self.config.connector_name, self.config.trading_pair
                )
                self._tick_size = float(rules.min_price_increment)
            except (KeyError, ValueError):  # pair not in the rules yet, or connector missing
                return None
        return self._tick_size

    @property
    def fee_maker(self) -> float:
        """Maker fee per fill in dollars. Fees are a fraction of notional, so this needs a price."""
        return float(self.config.fee_maker_pct) * float(self.current_mid or 0.0)

    @property
    def fee_taker(self) -> float:
        """Taker fee per unit in dollars, charged on the terminal liquidation."""
        return float(self.config.fee_taker_pct) * float(self.current_mid or 0.0)

    def _spreads(self, window: int | None = None) -> np.ndarray | None:
        """Touch spreads in dollars, optionally only the most recent `window` snapshots."""
        if not self._ob_snapshots:
            return None
        rows = list(self._ob_snapshots)
        if window:
            rows = rows[-window:]
        return np.fromiter(
            (row[1 + OB_DEPTH] - row[1] for row in rows), dtype=float, count=len(rows)
        )

    @property
    def median_spread(self) -> float | None:
        """
        Median touch spread over the most recent delta_window_snapshots -- the same window the
        model measures Delta over, recomputed live. The model's delta_ is frozen at the last fit,
        so comparing the two shows whether the spread has moved since.
        """
        spreads = self._spreads(self.config.delta_window_snapshots)
        return None if spreads is None else float(np.median(spreads))

    @property
    def median_spread_all(self) -> float | None:
        """
        Median over the whole sample. Not used by the model -- kept for comparison, since the gap
        between this and the windowed value is the reason for windowing at all.
        """
        spreads = self._spreads()
        return None if spreads is None else float(np.median(spreads))

    @property
    def drift_scale(self) -> float:
        """
        Linear ramp on the fitted drift, from drift_scale_start to drift_scale_end over the first
        drift_scale_trades market orders (sweeps merged). mu is estimated from realised price
        moves, so on a thin sample it is mostly noise -- and because the drift term scales with the
        horizon it can outweigh the half-spread many times over. Ramping it in keeps the early
        policy closer to neutral. Counted from the running MO counter, not the buffer, so it never
        runs back.
        """
        start, end = self.config.drift_scale_start, self.config.drift_scale_end
        span = self.config.drift_scale_trades
        if span <= 0:
            return end
        return start + (end - start) * min(self._mo_count / span, 1.0)

    def _maybe_fit_model(self):
        """
        Refit and re-solve at most once every model_refit_interval, once the evidence floor is met.

        The floor is evaluated on every attempt, fitted or not, so the status can show which
        regimes are still short and by how much. That replaces the old warm-up timer: the model
        starts when the data supports it, not when a clock says it probably does.

        Both steps run inline on the control loop. The solve is the expensive half -- a few
        hundred milliseconds at solver_dt=0.1 -- so this blocks the controller for that long
        every refit. Fine for prototyping; it belongs in an executor before this goes live.
        """
        now = self.market_data_provider.time()
        if now - self._last_fit_attempt < self.config.model_refit_interval:
            return
        self._last_fit_attempt = now

        book_features = self.book_features.transform_book(self.ob_snapshots_df)
        mos = collapse_sweeps(self.trades_df, self.config.sweep_window_seconds)
        stats = SufficientStatistics(
            self.regime_def,
            window_mos=self.config.window_mos,
            min_mos=self.config.min_mos_per_regime,
            min_sojourns=self.config.min_sojourns_per_regime,
        ).fit(book_features, mos)
        self._stats = stats
        tick = self.tick_size
        if not stats.ready_ or tick is None:
            return

        started = time.perf_counter()
        try:
            model = ImbalanceMarkovModel(
                self.regime_def, tick,
                delta_window=self.config.delta_window_snapshots,
            ).fit(stats, book_features)
            solver = AtTheTouchSolver(model, {
                "phi": self.config.phi,
                "varphi": self.config.varphi,
                "q_min": self.config.q_min,
                "q_max": self.config.q_max,
                "T": self.config.solver_horizon,
                "dt": self.config.solver_dt,
                "fee_maker": self.fee_maker,
                "fee_taker": self.fee_taker,
                "drift_scale": self.drift_scale,
            }).solve()
        except (ValueError, ArithmeticError) as e:
            # fit() and solve() signal an unmet precondition with ValueError (numpy's LinAlgError is
            # a ValueError too); ArithmeticError covers zero-division / overflow in the numerics.
            # The preconditions can still fail after the floor passes -- a regime whose only MOs on
            # one side landed within the eps horizon of the end of the sample has no eps yet. The
            # previous policy, if any, stays in force.
            error = f"{type(e).__name__}: {e}"
            if error != self._fit_error:
                self.logger().warning(f"{self._log_tag} refit failed, keeping the previous policy: {error}")
            self._fit_error = error
            return

        self.model, self.solver = model, solver
        self._last_fit_ts = now
        self._fit_seconds = time.perf_counter() - started
        self._fit_error = None

    # Trading -------------------------------------------------------------------------------

    @property
    def base_asset(self) -> str:
        return self.config.trading_pair.split("-")[0]

    @property
    def quote_asset(self) -> str:
        return self.config.trading_pair.split("-")[1]

    def _anchor_holding(self):
        """
        On spot, fix the offset between the exchange's base balance and our own fills, once.

        Read on the first tick, when the connector is ready and before any of our orders rest. The
        TOTAL balance is used, locked amounts included, so a resting sell does not read as a unit
        already gone. Subtracting the fills already on the books matters after a restart: the
        orchestrator reloads this controller's position from the database, and the balance
        already contains it, so without the subtraction it would be counted twice.

        After this, the holding moves only with our own fills -- not with the live balance, which
        updates on its own schedule (it would flicker by a unit around every fill) and which any
        deposit, withdrawal or other bot on the account would also move. The status compares the
        two so any drift, e.g. buy fees charged in the base asset, stays visible. A restart
        re-anchors.
        """
        if self._holding_offset is not None:
            return
        try:
            balance = Decimal(str(
                self.market_data_provider.get_balance(self.config.connector_name, self.base_asset)
            ))
        except ValueError:
            return  # connector not registered yet; retry next tick
        self._holding_offset = balance - self.net_position
        self.logger().info(
            f"{self._log_tag} anchored {self.base_asset} holding at {balance} (baseline "
            f"{self.config.baseline_holding}, inventory {self.inventory:+d} units)"
        )

    @property
    def holding(self) -> Decimal | None:
        """
        Base units held, as far as the model is concerned: our own netted fills plus the anchored
        offset. On a perpetual the offset is zero, so this is the netted position as before. None
        on spot until the anchor is taken.
        """
        if self._holding_offset is None:
            return None
        return self.net_position + self._holding_offset

    @property
    def deviation(self) -> Decimal | None:
        """Holding minus the baseline, in base units -- the inventory the policy manages."""
        holding = self.holding
        return None if holding is None else holding - self.config.baseline_holding

    @property
    def show_baseline(self) -> bool:
        """
        Whether the baseline lines belong in the status. Not on a perpetual with no baseline:
        there the holding and the deviation are the same thing and the status reads as before.
        """
        return not self.config.is_perpetual or self.config.baseline_holding != 0

    @property
    def inventory(self) -> int:
        """
        Deviation from the baseline in whole units of order_amount, rounded to the nearest unit.

        With a zero baseline on a perpetual this is the netted position in units, as before.
        Rounding is needed because min_base_amount_increment allows partial fills, while the
        model's grid is integer units; the nearest unit is close enough for a policy lookup.
        Reads 0 on spot before the anchor, but nothing is quoted then -- see intended_quotes.
        """
        deviation = self.deviation
        if self.config.order_amount <= 0 or deviation is None:
            return 0
        return round(float(deviation / self.config.order_amount))

    def _quantize(self, price: float) -> Decimal:
        """Round a price onto the exchange's tick grid, so live and intended prices compare exactly."""
        return self.market_data_provider.quantize_order_price(
            self.config.connector_name, self.config.trading_pair, Decimal(str(price))
        )

    def _live_quote(self, level_id: str) -> ExecutorInfo | None:
        """The active quote executor for one side, if there is one."""
        for executor in self.executors_info:
            if executor.is_active and executor.custom_info.get("level_id") == level_id:
                return executor
        return None

    def quote_plan(self) -> list[dict] | None:
        """
        Reconcile the live quotes against what the policy wants, one entry per side.

        The rule that matters: a live order already sitting at the intended price is left alone.
        Cancelling and replacing at the same level would surrender queue position for nothing --
        at the touch, queue position is most of the edge. So a quote is only replaced when the
        touch has actually moved away from it.

        action is one of:
            keep     -- live order is already at the intended price, do nothing
            place    -- policy wants this side and nothing is live
            replace  -- live order is at a stale price, cancel it and post at the touch
            cancel   -- policy no longer wants this side
            idle     -- policy does not want this side and nothing is live
        """
        quotes = self.intended_quotes()
        if quotes is None:
            return None

        plan = []
        for level_id, side, wanted, price in (
            (LEVEL_BID, TradeType.BUY,
             quotes["post_bid"] and self.config.enable_bid
             and self.cooloff_remaining(LEVEL_BID) <= 0.0, quotes["bid_price"]),
            (LEVEL_ASK, TradeType.SELL,
             quotes["post_ask"] and self.config.enable_ask
             and self.cooloff_remaining(LEVEL_ASK) <= 0.0, quotes["ask_price"]),
        ):
            live = self._live_quote(level_id)
            live_price = getattr(live.config, "price", None) if live else None
            cooloff = self.cooloff_remaining(level_id)
            if not wanted:
                action = "cancel" if live else "idle"
            elif live is None:
                action = "place"
            elif live_price == price:
                action = "keep"
            else:
                action = "replace"
            plan.append({
                "level_id": level_id, "side": side, "action": action,
                "intended_price": price, "live_price": live_price,
                "live_id": live.id if live else None, "amount": quotes["amount"],
                "cooloff": cooloff,
            })
        return plan

    def _actions_from_plan(self, plan: list[dict]) -> list[ExecutorAction]:
        """Stops first, so a stale quote is cancelled before its replacement is posted."""
        stops, creates = [], []
        for entry in plan:
            if entry["action"] in ("cancel", "replace"):
                stops.append(StopExecutorAction(
                    controller_id=self.config.id,
                    executor_id=entry["live_id"],
                    keep_position=True,
                ))
            if entry["action"] in ("place", "replace"):
                creates.append(CreateExecutorAction(
                    controller_id=self.config.id,
                    executor_config=OrderExecutorConfig(
                        timestamp=self.market_data_provider.time(),
                        controller_id=self.config.id,
                        connector_name=self.config.connector_name,
                        trading_pair=self.config.trading_pair,
                        side=entry["side"],
                        amount=entry["amount"],
                        # LIMIT_MAKER only. If the touch moves before the order lands it is
                        # rejected rather than crossing, which is the desired outcome -- we do not
                        # retry within the tick. The next loop reprices against the new touch.
                        execution_strategy=ExecutionStrategy.LIMIT_MAKER,
                        price=entry["intended_price"],
                        leverage=self.config.leverage,
                        level_id=entry["level_id"],
                    ),
                ))
        return stops + creates

    def intended_quotes(self) -> dict | None:
        """
        What the policy wants right now: whether to post each side, and at what price.

        Always the current touch -- there is no refresh tolerance, so a live quote is replaced
        whenever the touch moves. Returns None until a policy exists.

        Note: time_remaining is always the full horizon, so the policy used is the quasi-stationary
        one (it is flat from T down to roughly the last minute). That means the terminal condition
        is never actually reached and there is no end-of-session flatten. If one is wanted later,
        it belongs here: feed a real countdown into should_post, or add explicit unwind logic.
        """
        if self.solver is None or not self._ob_snapshots:
            return None
        if self.holding is None:
            return None     # spot, not anchored yet: inventory is unknown, so quote nothing
        regime = self.current_regime
        if regime is None:
            return None

        last = self._ob_snapshots[-1]
        best_bid, best_ask = self._quantize(last[1]), self._quantize(last[1 + OB_DEPTH])
        q = self.inventory
        q_clamped = min(max(q, self.config.q_min), self.config.q_max)
        post_bid, post_ask = self.solver.should_post(
            self.config.solver_horizon, regime, q_clamped
        )
        return {
            "regime": regime,
            "q": q,
            "q_clamped": q_clamped,
            "post_bid": post_bid,
            "post_ask": post_ask,
            "bid_price": best_bid,
            "ask_price": best_ask,
            "amount": self.config.order_amount,
        }

    @override
    def determine_executor_actions(self) -> list[ExecutorAction]:
        plan = self.quote_plan()
        if plan is None:
            return []
        return self._actions_from_plan(plan)

    def create_actions_proposal(self) -> list[ExecutorAction]:
        raise NotImplementedError

    def stop_actions_proposal(self) -> list[ExecutorAction]:
        raise NotImplementedError

    @property
    def net_position(self) -> Decimal:
        """Signed net position in base units, from the exchange's netted position."""
        signed = Decimal(0)
        for position in self.positions_held:
            if (position.connector_name == self.config.connector_name
                    and position.trading_pair == self.config.trading_pair):
                signed += position.amount if position.side == TradeType.BUY else -position.amount
        return signed

    @property
    def current_mid(self) -> Decimal | None:
        """Midprice from the latest snapshot, as a Decimal so it composes with position sizes."""
        if not self._ob_snapshots:
            return None
        last = self._ob_snapshots[-1]
        return (Decimal(str(last[1])) + Decimal(str(last[1 + OB_DEPTH]))) / 2

    def _track_fills(self):
        """
        Stamp a fill time for whichever side moved the position since the last tick.

        A bid fill raises the net position and an ask fill lowers it. The one blind spot is a bid
        and an ask filling inside the same tick, which would net out -- rare with one order per
        side, and the only cost is a missed cooloff.
        """
        current = self.net_position
        if self._last_position is not None and current != self._last_position:
            now = self.market_data_provider.time()
            side = "bid" if current > self._last_position else "ask"
            if side == "bid":
                self._last_bid_fill_ts = now
            else:
                self._last_ask_fill_ts = now
            # Notional at the mid rather than the fill price: the two differ by half a spread at
            # most, which is noise next to what the counter is for.
            mid = self.current_mid
            notional = float(abs(current - self._last_position) * mid) if mid is not None else 0.0
            for counters in (self._interval, self._totals):
                counters[f"fills_{side}"] += 1
                counters[f"volume_{side}"] += notional
        self._last_position = current

    @staticmethod
    def _empty_activity() -> dict:
        return {
            "fills_bid": 0, "fills_ask": 0, "volume_bid": 0.0, "volume_ask": 0.0,
            "live_bid_s": 0.0, "live_ask_s": 0.0, "elapsed_s": 0.0,
        }

    def _track_activity(self):
        """
        Credit the time since the last tick to each side that has a quote resting now. Sampled
        per tick, so it is exact to within one control interval.
        """
        now = self.market_data_provider.time()
        last, self._last_activity_ts = self._last_activity_ts, now
        if last is None:
            return
        dt = now - last
        for counters in (self._interval, self._totals):
            counters["elapsed_s"] += dt
            if self._live_quote(LEVEL_BID) is not None:
                counters["live_bid_s"] += dt
            if self._live_quote(LEVEL_ASK) is not None:
                counters["live_ask_s"] += dt

    def cooloff_remaining(self, level_id: str) -> float:
        """Seconds left before this side may quote again, 0.0 if it is free to quote."""
        last = self._last_bid_fill_ts if level_id == LEVEL_BID else self._last_ask_fill_ts
        if last is None:
            return 0.0
        elapsed = self.market_data_provider.time() - last
        return max(0.0, self.config.fill_cooldown_seconds - elapsed)

    # Status -------------------------------------------------------------------------------
    #
    # Ordered by what a live operator needs first: what we are quoting and at what edge, then
    # position, then whether the model behind it is healthy, and the full posting policy last as
    # reference. Each fact appears exactly once. Everything price-dimensioned is shown in TICKS,
    # which is the only scale-free unit here -- dollars are meaningless across a $0.11 and a
    # $60,000 asset, and bps are given alongside where the comparison is to a fee.

    def _fmt_px(self, price) -> str:
        """A price at the tick's own precision, so 0.11320000000000001 reads as 0.11320."""
        tick = self.tick_size
        if not tick:
            return f"{float(price):.8g}"
        exponent = Decimal(str(tick)).as_tuple().exponent
        decimals = max(0, -exponent) if isinstance(exponent, int) else 0
        return f"{float(price):.{decimals}f}"

    def _t(self, value: float) -> str:
        """A price-dimensioned value in ticks."""
        if not bool(np.isfinite(value)):
            return "inf"
        tick = self.tick_size
        return "?t" if not tick else f"{value / tick:.2f}t"

    def _ts(self, value: float) -> str:
        """Same, but always signed -- for terms that are summed, where the sign is the point."""
        if not bool(np.isfinite(value)):
            return "inf"
        tick = self.tick_size
        return "?t" if not tick else f"{value / tick:+.2f}t"

    def _bps(self, value: float, reference: float | None = None) -> str:
        ref = float(reference if reference is not None else (self.current_mid or 0))
        return "? bps" if ref <= 0 else f"{value / ref * 1e4:.2f} bps"

    @override
    def to_format_status(self) -> list[str]:
        lines = [
            "",
            "=" * 118, (
                f"  mm_at_the_touch (market making at the touch)  |  {self.config.connector_name}"
                f"  |  {self.config.trading_pair}"
            ),
            "=" * 118,
        ]
        lines.extend(self._market_lines())
        lines.extend(self._quote_lines())
        lines.extend(self._position_lines())
        lines.extend(self._model_lines())
        return lines

    def _market_lines(self) -> list[str]:
        """
        The book as it stands, plus the one spread the model is built on.

        Two different spreads matter: the LIVE spread (this instant's touch) and Delta (the median
        over delta_window_snapshots, which the model freezes at each fit). At the touch Delta/2 is
        the edge the solver credits every fill with, so it is named rather than left to be
        mistaken for the live one.
        """
        lines = ["  MARKET"]
        if not self._ob_snapshots:
            return lines + ["    no order book snapshots yet"]

        last = self._ob_snapshots[-1]
        bid, ask = last[1], last[1 + OB_DEPTH]
        mid, spread = (bid + ask) / 2, ask - bid
        age = self.market_data_provider.time() - last[0]
        lines.append(
            f"    mid {self._fmt_px(mid)}   touch {self._fmt_px(bid)} / {self._fmt_px(ask)}"
            f"   live spread {self._t(spread)} ({self._bps(spread, mid)})"
            f"   [snapshot {age:.1f}s old]"
        )

        delta = self.median_spread
        if delta is not None:
            whole = self.median_spread_all
            note = ""
            if whole is not None and abs(whole - delta) > (self.tick_size or 0) / 2:
                note = f"   (whole sample {self._t(whole)})"
            lines.append(
                f"    Delta {self._t(delta)} ({self._bps(delta, mid)})"
                f" = median of last {self.config.delta_window_snapshots} snapshots"
                f"  ->  Delta/2 is the edge the solver credits each fill{note}"
            )

        # A fill at the touch earns Delta/2 and pays the maker fee, so a round trip -- one fill on
        # each side -- has to clear two maker fees before it earns anything.
        rt = 2.0 * float(self.config.fee_maker_pct) * mid
        lines.append(
            f"    fees  maker {float(self.config.fee_maker_pct) * 100:.4f}%"
            f"   taker {float(self.config.fee_taker_pct) * 100:.4f}%"
            f"   |  maker round trip (2 fills) {self._t(rt)} ({self._bps(rt, mid)})"
            f"   |  live spread covers it: {'YES' if spread > rt else 'NO'}"
        )

        trade = self.processed_data.get("last_trade")
        tape = (
            "no public trade yet" if trade is None else
            f"last {trade.type.name} {trade.amount} @ {self._fmt_px(trade.price)}"
            f" ({self.market_data_provider.time() - trade.timestamp:.1f}s ago)"
        )
        lines.append(
            f"    data  {len(self._ob_snapshots)} snapshots / {len(self._trades)} prints"
            f" = {self._mo_count} market orders (sweeps merged)"
            f"   |  {tape}"
        )
        return lines

    def _quote_lines(self) -> list[str]:
        """
        What we are quoting right now, and whether it pays.

        At the touch the price is never in question -- it is always the best bid/ask -- so the
        table is about the decision: whether each side posts, and if not, why not. The reasons are
        told apart because they call for different responses: the policy judging the edge not
        worth it, the inventory band barring the side, or the side being disabled in the config.
        """
        lines = ["", "  QUOTES"]
        enabled = [n for n, on in (("bid", self.config.enable_bid),
                                   ("ask", self.config.enable_ask)) if on]
        if len(enabled) < 2:
            lines.append(
                f"    sides enabled: {', '.join(enabled) if enabled else 'NONE (observation mode)'}"
            )

        quotes = self.intended_quotes()
        if quotes is None:
            return lines + ["    no policy solved yet, so nothing is being quoted"]

        regime = self.regime_def.regime_labels[quotes["regime"]]
        if quotes["q"] != quotes["q_clamped"]:
            lines.append(
                f"    WARNING inventory {quotes['q']} is outside the solved band"
                f" [{self.config.q_min}, {self.config.q_max}]; policy read at"
                f" {quotes['q_clamped']}"
            )
        lines.append(f"    regime {regime}   |  policy read at q = {quotes['q_clamped']}")

        plan = {e["level_id"]: e for e in (self.quote_plan() or [])}
        detail = {
            "keep": "already at the touch, left alone (keeps queue position)",
            "place": "posting LIMIT_MAKER",
            "replace": "touch moved, cancelling and reposting",
            "cancel": "cancelling",
            "idle": "standing aside",
        }
        lines.append(f"    {'side':5s} {'price':>11s}  {'action':8s} note")
        for level_id, name, post, side_enabled, barred in (
            (LEVEL_BID, "BID", quotes["post_bid"], self.config.enable_bid,
             quotes["q_clamped"] >= self.config.q_max),
            (LEVEL_ASK, "ASK", quotes["post_ask"], self.config.enable_ask,
             quotes["q_clamped"] <= self.config.q_min),
        ):
            entry = plan.get(level_id)
            if entry is None:
                continue
            action = entry["action"]
            note = detail[action]
            if action in ("cancel", "idle"):
                if barred:
                    note += " -- inventory band bars this side"
                elif not post:
                    note += " -- policy says the edge is not worth it"
                elif not side_enabled:
                    note += " -- side disabled in config"
            if entry["live_price"] is not None and action in ("keep", "replace", "cancel"):
                note += f"  [live {self._fmt_px(entry['live_price'])}]"
            if entry["cooloff"] > 0:
                note += f"  [COOLOFF {entry['cooloff']:.1f}s]"
            price = "-" if action in ("cancel", "idle") else self._fmt_px(entry["intended_price"])
            lines.append(f"    {name:5s} {price:>11s}  {action.upper():8s} {note}")
        lines.extend(self._edge_lines(quotes, plan))
        return lines

    def _edge(self, quotes: dict, plan: dict) -> dict | None:
        """
        The inputs to the edge calculation for the current regime, in dollars per base unit and
        fills per hour. Shared by the status and the status log so the two cannot disagree.
        """
        m = self.model
        if m is None:
            return None
        z = quotes["regime"]
        fee = self.fee_maker
        bid, ask = float(quotes["bid_price"]), float(quotes["ask_price"])
        eps_ask, eps_bid = float(m.eps_plus_[z]), float(m.eps_minus_[z])
        # Economically "quoting this side" means we intend a quote there, which includes the tick
        # a replace spends between the cancel and the repost.
        on = {k: (v["action"] not in ("idle", "cancel")) for k, v in plan.items()}
        return {
            "fee": fee,
            "mid": (bid + ask) / 2.0,
            "spread": ask - bid,
            "eps_ask": eps_ask,
            "eps_bid": eps_bid,
            "rate_ask": float(m.lam_plus_[z]) * 3600.0,
            "rate_bid": float(m.lam_minus_[z]) * 3600.0,
            "unit": float(self.config.order_amount),
            "on_bid": on.get(LEVEL_BID, False),
            "on_ask": on.get(LEVEL_ASK, False),
            "breakeven": 2 * fee + eps_ask + eps_bid,
        }

    def _edge_lines(self, quotes: dict, plan: dict) -> list[str]:
        """
        Whether the touch spread clears the maker fee round trip, and what it earns per hour.

        At the touch our spread IS the market spread, so a round trip captures it, pays the maker
        fee twice, and gives up the expected adverse move on both fills (eps+ + eps- in the
        current regime). Per side that is

            ask leg = Delta/2 - eps+ - fee        bid leg = Delta/2 - eps- - fee

        which is the posting gain c+- before the inventory term the solver adds on top.

        The model fills us on every matching MO, so the fill rate is just the MO arrival rate. That
        ignores the queue ahead of us at the touch, so fills/h and $/h are upper bounds.
        """
        e = self._edge(quotes, plan)
        if e is None:
            return []
        fee, mid, spread, unit = e["fee"], e["mid"], e["spread"], e["unit"]
        eps_ask, eps_bid, rate_ask, rate_bid = e["eps_ask"], e["eps_bid"], e["rate_ask"], e["rate_bid"]
        on_bid, on_ask, breakeven = e["on_bid"], e["on_ask"], e["breakeven"]

        if not on_bid and not on_ask:
            return [("    edge  nothing quoted, so no spread is being captured"
                     f"   |  breakeven spread would be {self._t(breakeven)}"
                     f" ({self._bps(breakeven, mid)})")]

        if not (on_bid and on_ask):
            # One-sided: the closing leg happens at a price that does not exist yet, so there is
            # no round trip to price. Report the live leg on its own.
            side = "ASK only (selling)" if on_ask else "BID only (buying)"
            eps, rate = (eps_ask, rate_ask) if on_ask else (eps_bid, rate_bid)
            leg = spread / 2.0 - eps - fee
            return [
                f"    edge  {side} -- one-sided, so no round trip completes",
                (
                    f"          this leg  half-spread {self._ts(spread / 2.0)}  adverse {self._ts(-eps)}"
                    f"  fee {self._ts(-fee)}  =  {self._ts(leg)} ({self._bps(leg, mid)})"
                    f"  = ${leg * unit:+.5f}/fill at up to {rate:.1f} fills/h"
                )
            ]

        net = spread - 2 * fee - eps_ask - eps_bid
        # A round trip needs one fill on each side, so the slower side sets the pace.
        round_trips = min(rate_ask, rate_bid)
        per_hour = round_trips * net * unit

        if net <= 0:
            verdict = "LOSS per round trip at these rates -- the spread cannot support it"
        elif round_trips < 0.5:
            verdict = ("edge is positive but market orders on the slow side arrive too rarely"
                       " for round trips to complete")
        else:
            verdict = f"viable: up to {round_trips:.1f} round trips/h at ${net * unit:+.5f} each"

        return [
            (
                f"    edge  spread {self._t(spread)} ({self._bps(spread, mid)})"
                f"   |  breakeven {self._t(breakeven)} ({self._bps(breakeven, mid)})"
            ),
            (
                f"          round trip  gross {self._ts(spread)}   fees {self._ts(-2 * fee)}"
                f"   adverse {self._ts(-(eps_ask + eps_bid))}"
                f"   =  NET {self._t(net)} ({self._bps(net, mid)}) = ${net * unit:+.5f}/unit"
            ),
            (
                f"          fills/h  bid {rate_bid:.1f}   ask {rate_ask:.1f}"
                f"   ->  {round_trips:.1f} round trips/h  =  ${per_hour:+.4f}/h"
                f"   (upper bound: assumes no queue ahead of us)"
            ),
            f"    VERDICT  {verdict}",
        ]

    def _position_lines(self) -> list[str]:
        """
        Position and P&L.

        The inventory the policy is indexed by is the deviation from baseline_holding, in units.
        With a zero baseline on a perpetual that is simply the netted position, and this block
        reads as it always has. Otherwise it also shows the holding behind the deviation and, on
        spot, a check of that holding against the exchange balance. P&L is the exchange
        position's own figure for this pair, which covers our fills only, not the baseline.
        """
        base = self.base_asset
        mid = self.current_mid
        lines = ["", "  POSITION"]
        holding, raw = self.holding, self.deviation
        if holding is None or raw is None:
            return lines + [
                f"    waiting to read the {base} balance that anchors inventory; nothing is quoted"
            ]

        units = self.inventory
        state = "FLAT" if raw == 0 else ("LONG" if raw > 0 else "SHORT")
        bound = ""
        if units >= self.config.q_max:
            bound = "   <- at q_max, the BID is barred"
        elif units <= self.config.q_min:
            bound = "   <- at q_min, the ASK is barred"
        vs = ""
        if self.show_baseline:
            vs = (f" vs baseline {float(self.config.baseline_holding):g}"
                  f" (holding {float(holding):g})")
        leverage = f"   |  leverage {self.config.leverage}" if self.config.is_perpetual else ""
        lines.append(
            f"    {state} {float(raw):+g} {base}{vs} = {units:+d} of"
            f" [{self.config.q_min:+d}, {self.config.q_max:+d}]"
            f"   |  {'n/a' if mid is None else f'${float(raw * mid):+.2f}'} notional"
            f"   |  1 unit = {self.config.order_amount} {base}{leverage}{bound}"
        )
        if not self.config.is_perpetual:
            lines.extend(self._spot_balance_lines(holding, units, mid))

        pnl = sum((p.global_pnl_quote for p in self._pair_positions()), Decimal(0))
        lines.append(f"    pnl ${float(pnl):+.4f} (exchange, this pair)")
        return lines

    def _pair_positions(self) -> list:
        """The exchange positions held for this connector and pair."""
        return [
            p for p in self.positions_held
            if p.connector_name == self.config.connector_name
            and p.trading_pair == self.config.trading_pair
        ]

    def _balances(self) -> tuple[Decimal, Decimal] | None:
        """Total base and quote balances on the exchange, or None if they cannot be read."""
        cn = self.config.connector_name
        try:
            return (
                Decimal(str(self.market_data_provider.get_balance(cn, self.base_asset))),
                Decimal(str(self.market_data_provider.get_balance(cn, self.quote_asset))),
            )
        except (ValueError, ArithmeticError):
            # ValueError: the connector is not registered; ArithmeticError: Decimal rejected the
            # balance string (InvalidOperation).
            return None

    def _spot_balance_lines(self, holding: Decimal, units: int, mid: Decimal | None) -> list[str]:
        """
        The modelled holding against the exchange's, plus whether the quote balance can fund the
        buys up to q_max.

        Drift between the two holdings means something moved the balance that our fills did not
        account for: buy fees charged in the base asset, a deposit or withdrawal, another bot on
        the account. A restart re-anchors. The quote check is a warning, not a gate -- an unfunded
        bid is simply rejected by the exchange, which is a safe failure on spot.
        """
        balances = self._balances()
        if balances is None:
            return ["    balance  exchange balances not readable right now"]
        exchange, quote = balances
        line = (
            f"    balance  exchange {float(exchange):g} {self.base_asset} vs modelled"
            f" {float(holding):g} (drift {float(exchange - holding):+g})"
            f"   |  {float(quote):.2f} {self.quote_asset}"
        )
        if mid is not None:
            need = max(self.config.q_max - units, 0) * self.config.order_amount * mid
            line += f", buys up to q_max need ~{float(need):.2f}"
            if quote < need:
                line += "  <- SHORT of quote, top bids will be rejected"
        return [line]

    def _model_lines(self) -> list[str]:
        """
        Whether the fit behind the policy is healthy, then the policy itself as reference.

        This block is last on purpose: it is what you read when something upstream looks wrong,
        not what you watch tick to tick.
        """
        lines = ["", "  MODEL"]
        stats = self._stats
        if stats is None:
            return lines + ["    gathering data, model not evaluated yet"]

        blockers = self._blockers(stats)
        if blockers:
            lines.append(f"    WAITING on {len(blockers)} condition(s):")
            lines.extend(f"      - {b}" for b in blockers)
        if self._fit_error:
            lines.append(f"    last fit error: {self._fit_error}")
        lines.extend(self._evidence_lines(stats))
        if self.solver is None or self.model is None:
            return lines

        m, sol = self.model, self.solver
        age = self.market_data_provider.time() - (self._last_fit_ts or 0.0)
        edge = m.delta_ / 2.0 - self.fee_maker
        lines.append(
            f"    fitted {age:.0f}s ago in {self._fit_seconds:.2f}s"
            f"   |  horizon T={self.config.solver_horizon:g}s dt={self.config.solver_dt:g}s"
            f"   |  Delta at fit {self._t(m.delta_)}, so Delta/2 - maker fee = {self._ts(edge)}"
            " per fill"
        )
        ramp_left = max(0, self.config.drift_scale_trades - self._mo_count)
        lines.append(
            f"    drift   mu damped to {sol.drift_scale:.1%} of the fitted value"
            f" (ramping {self.config.drift_scale_start:g} -> {self.config.drift_scale_end:g}"
            f" over {self.config.drift_scale_trades} market orders, {ramp_left} to go)"
        )

        lines.append("")
        lines.extend(self._regime_table_lines(m))
        lines.append("")
        lines.extend(self._posting_grid_lines(sol))
        return lines

    def _evidence_lines(self, stats: "SufficientStatistics") -> list[str]:
        """
        What each regime's window holds and how far back it reaches. 'reaches back' is the age of
        the oldest visit used; 'floor' marks a regime the evidence floor pushed past the global
        window. A regime that stays at 'floor' for hours is the one whose estimates are stale.
        """
        now = self.market_data_provider.time()
        lines = [
            (
                f"    evidence per regime  (window: last {self.config.window_mos or 'all'} market orders;"
                f" floor {stats.min_mos} buy + {stats.min_mos} sell orders and"
                f" {stats.min_sojourns} completed visits)"
            ),
            f"      {'regime':22s} {'visits':>7s} {'buys':>6s} {'sells':>6s}  {'reaches back':>12s}",
        ]
        for label, row in stats.coverage_.to_dict("index").items():
            back = "-" if np.isnan(row["window_start"]) else f"{(now - row['window_start']) / 60.0:.1f}m"
            note = "floor" if row["extended"] else ""
            if not row["met"]:
                note = "SHORT"
            lines.append(
                f"      {label:22s} {int(row['visits']):>7d} {int(row['buy_mos']):>6d}"
                f" {int(row['sell_mos']):>6d}  {back:>12s}  {note}"
            )
        return lines

    def _regime_table_lines(self, m: "ImbalanceMarkovModel") -> list[str]:
        """
        The fitted per-regime parameters, with standard errors beside eps and mu.

        eps, edge and mu are in TICKS rather than dollars: on a sub-dollar asset the dollar values
        are all zeroes to four places, and ticks is the unit the rest of the status uses. mu is per
        hour for the same reason -- per second it is indistinguishable from zero.

        edge is Delta/2 - eps - maker fee, the posting gain before the inventory term. Where it is
        negative the policy only posts when inventory pressure makes up the difference.

        The standard errors assume independent MOs, so they are optimistic (see mu_se_). Their
        use is relative: a mu within about two se of zero is not a directional signal, and an eps
        se that is large next to eps says the window is too small for that regime.
        """
        tick = self.tick_size or float("nan")
        half = m.delta_ / 2.0 - self.fee_maker
        table = pd.DataFrame({
            "dtau(s)": m.stats_.delta_tau_,
            "1/Lam(s)": 1.0 / m.Lambda_,
            "lam+/s": m.lam_plus_,
            "lam-/s": m.lam_minus_,
            "eps+(t)": m.eps_plus_ / tick,
            "se": m.eps_plus_se_ / tick,
            "eps-(t)": m.eps_minus_ / tick,
            "se ": m.eps_minus_se_ / tick,
            "edge+(t)": (half - m.eps_plus_) / tick,
            "edge-(t)": (half - m.eps_minus_) / tick,
            "mu(t/h)": m.mu_ * 3600.0 / tick,
            "se  ": m.mu_se_ * 3600.0 / tick,
        }, index=self.regime_def.regime_labels)
        return ["    " + ln for ln in table.round(3).to_string().split("\n")]

    def _posting_grid_lines(self, sol: "AtTheTouchSolver") -> list[str]:
        """
        The whole policy: where the agent posts, per regime and inventory, at the start of the
        horizon.

        Two rows per regime, 'a' for the ask and 'b' for the bid. '--' is a side the inventory
        band bars (the ask at q_min, the bid at q_max). Read along a row to see where inventory
        pressure switches a side on.
        """
        lines = [
            "    POLICY -- where to post (A / B = post ask / bid, . = stand aside, -- = barred)",
            f"    {'regime':22s}  {'q':>4s}" + "".join(f"{v:>4d}" for v in sol.q_),
        ]
        last = len(sol.q_) - 1
        for i, label in enumerate(self.regime_def.regime_labels):
            for tag, grid, mark, barred in (("a", sol.ell_plus_, "A", 0),
                                            ("b", sol.ell_minus_, "B", last)):
                cells = "".join(
                    "  --" if j == barred else f"{mark if post else '.':>4s}"
                    for j, post in enumerate(grid[0, i])
                )
                lines.append(f"    {label if tag == 'a' else '':22s}  {tag:>4s}{cells}")
        return lines

    # Status log ----------------------------------------------------------------------------
    #
    # One key=value line per pair every status_log_interval seconds, for an agent that reads the
    # log rather than the screen. Only what its decisions need: the API keeps few lines per bot,
    # so every field here has to earn its place. The full detail stays on the status screen.

    def _maybe_log_status(self):
        interval = self.config.status_log_interval
        now = self.market_data_provider.time()
        if interval <= 0 or now - self._last_status_log < interval:
            return
        self._last_status_log = now
        self.logger().info(self._status_line())
        self._interval = self._empty_activity()

    @property
    def _log_tag(self) -> str:
        """Every log line names its pair: controllers share one logger, so nothing else does."""
        return f"mm_at_the_touch pair={self.config.trading_pair}"

    @property
    def _state(self) -> str:
        if self.config.enable_bid and self.config.enable_ask:
            return "active"
        return "standby" if not (self.config.enable_bid or self.config.enable_ask) else "partial"

    def _shortfall(self) -> tuple[int, str]:
        """Regimes still short of the evidence floor, and on what: mos, visits, both, or '-'."""
        stats = self._stats
        if stats is None:
            return 0, "-"
        short = stats.coverage_[~stats.coverage_["met"]]
        mos = bool(((short["buy_mos"] < stats.min_mos) | (short["sell_mos"] < stats.min_mos)).any())
        visits = bool((short["visits"] < stats.min_sojourns).any())
        on = "both" if mos and visits else "mos" if mos else "visits" if visits else "-"
        return len(short), on

    def _economics(self) -> tuple[float, float] | None:
        """
        In the current regime, at true fees: edge per fill in bps, half of (Delta - breakeven) over
        the mid, and the volume per hour we could do quoting both sides, est_fills x unit notional.
        est_fills is twice the slower side's market-order rate, capped by what the cooloff allows.
        """
        m, regime, mid, spread = self.model, self.current_regime, self.current_mid, self.median_spread
        if m is None or regime is None or mid is None or spread is None:
            return None
        mid = float(mid)
        breakeven = 2.0 * self.fee_maker + float(m.eps_plus_[regime]) + float(m.eps_minus_[regime])
        edge_bps = (spread - breakeven) / 2.0 / mid * 1e4
        rate = min(float(m.lam_plus_[regime]), float(m.lam_minus_[regime])) * 3600.0
        cap = 3600.0 / max(self.config.fill_cooldown_seconds, 1.0)
        fills_h = 2.0 * min(rate, cap)
        return edge_bps, fills_h * float(self.config.order_amount) * mid

    def _status_line(self) -> str:
        """
        state     active | standby | partial (one side enabled)
        fit       1 once a policy exists. short=regimes below the evidence floor, short_on=what
                  they lack (mos/visits/both), floor=min_mos_per_regime, err=1 if the last refit failed
        q, post   inventory in units, and the sides the policy wants now (B, A, BA or -)
        edge_bps, est_vol_h   see _economics; nan before the first fit
        vol, fills, live_bid_s, live_ask_s, up_s   session totals: own filled notional, fill count,
                  seconds a quote rested per side, seconds observed. Difference them between reads.
        vol_h     the last interval's own volume as an hourly rate
        pnl, upnl, fees       net PnL incl. unrealized, the unrealized part, and fees, in quote
        age_s     seconds since the last order book snapshot
        """
        now = self.market_data_provider.time()
        short, short_on = self._shortfall()
        quotes = self.intended_quotes()
        post = "-" if quotes is None else (
            ("B" if quotes["post_bid"] else "") + ("A" if quotes["post_ask"] else "") or "-"
        )
        econ = self._economics()
        edge, est = ("nan", "nan") if econ is None else (f"{econ[0]:.2f}", f"{econ[1]:.0f}")
        i, t = self._interval, self._totals
        vol_h = (i["volume_bid"] + i["volume_ask"]) * 3600.0 / i["elapsed_s"] if i["elapsed_s"] > 0 else 0.0
        positions = self._pair_positions()

        def total(attr: str) -> float:
            return float(sum((getattr(p, attr) for p in positions), Decimal(0)))

        age = f"{now - self._ob_snapshots[-1][0]:.1f}" if self._ob_snapshots else "nan"
        return (
            f"{self._log_tag} status state={self._state} fit={int(self.solver is not None)}"
            f" short={short} short_on={short_on} floor={self.config.min_mos_per_regime}"
            f" err={int(self._fit_error is not None)}"
            f" q={self.inventory:+d} post={post}"
            f" edge_bps={edge} est_vol_h={est}"
            f" vol={t['volume_bid'] + t['volume_ask']:.2f} fills={t['fills_bid'] + t['fills_ask']}"
            f" live_bid_s={t['live_bid_s']:.0f} live_ask_s={t['live_ask_s']:.0f} up_s={t['elapsed_s']:.0f}"
            f" vol_h={vol_h:.0f}"
            f" pnl={total('global_pnl_quote'):.4f} upnl={total('unrealized_pnl_quote'):.4f}"
            f" fees={total('cum_fees_quote'):.4f} age_s={age}"
        )

    # Status helpers ------------------------------------------------------------------------

    def _blockers(self, stats: "SufficientStatistics") -> list[str]:
        """Plain-language list of the conditions still unmet, with the specifics."""
        out = list(stats.blockers_)
        if self.tick_size is None:
            out.append("tick size not read from the connector's trading rules yet")
        return out

    @property
    def current_regime(self) -> int | None:
        """Regime index implied by the most recent snapshot, or None before the first one."""
        if not self._ob_snapshots:
            return None
        last = self._ob_snapshots[-1]
        levels = self.config.imbalance_levels
        bid = float(np.nansum(last[1 + 2 * OB_DEPTH:1 + 2 * OB_DEPTH + levels]))
        ask = float(np.nansum(last[1 + 3 * OB_DEPTH:1 + 3 * OB_DEPTH + levels]))
        if bid + ask <= 0.0:
            return None
        imbalance = (bid - ask) / (bid + ask)
        bounds = self.regime_def.imbalance_bounds
        idx = int(np.searchsorted(bounds, imbalance, side="left")) - 1
        return min(max(idx, 0), self.regime_def.n_regimes - 1)
