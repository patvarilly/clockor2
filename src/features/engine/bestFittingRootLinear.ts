import { rerootAndScale } from "./bestFittingRoot";
import { Tree, readNewick, writeNewick } from "phylojs";

// O(N) Best-Fitting Root (BFR) algorithm.
//
// clockor2's original BFR search (globalRootParallel) is O(N^2): for each of O(N)
// candidate edges, it reroots the tree (O(N)), recomputes all root-to-tip distances
// (O(N)), and evaluates the OLS regression.  This is fine for small trees but becomes
// a bottleneck for large ones.
//
// This implementation reduces the cost to O(N) by precomputing per-node subtree
// statistics in two DFS passes, then evaluating each candidate root position in O(1).
// TreeTime from the Neher lab (doi:10.1093/ve/vex042, Section 2.3 and Appendix; see
// treeregression.py) implements a more general weighted (GLS) version that accounts
// for phylogenetic correlations and larger mutation-count variances on longer branches.
// The algorithm here is the simpler OLS special case --- each tip's root-to-tip distance
// is considered independent and with equal weight --- which is exactly the optimization
// that clockor2's globalRootParallel performs.  This implementation was inspired by a
// similar one in the Delphy project
// (https://github.com/broadinstitute/delphy/blob/main/plans/2026-05-18-01-better-tree-init-round4-regression-rooting.md).
//
// === Subtree statistics ===
//
// For any node X, define three sets of statistics, all with distances measured from X:
//
//   - downstream_stats_at[X] = stats for tips downstream of X (including X itself if
//     a tip)
//   - upstream_stats_at[X] = stats for tips upstream of X
//   - all_stats_at[X] = stats for ALL tips = downstream_stats_at[X] + upstream_stats_at[X]
//
// Each is a RootingStats struct with five fields:
//
//   n        = number of tips
//   sum_dt   = sum of dt_i                    (dt_i = t_i - mean_t, centered dates)
//   sum_d    = sum of dist(X, tip_i)
//   sum_d_dt = sum of dist(X, tip_i) * dt_i
//   sum_d2   = sum of dist(X, tip_i)^2
//
// At the root R: upstream_stats_at[R] = {n:0, sum_dt:0, sum_d:0, sum_d_dt:0, sum_d2:0}
// (there are no tips upstream of the root).
//
// At a tip X: downstream_stats_at[X] = {n:1, sum_dt:dt_X, sum_d:0, sum_d_dt:0, sum_d2:0}
// (X is its own only downstream tip, at distance 0).
//
// === Three operations ===
//
// shift(s, L): shift the reference point away by L, increasing all distances by L.
// If r = shift(s, L):
//
//   r.n        = s.n
//   r.sum_dt   = s.sum_dt
//   r.sum_d    = L * s.n      + s.sum_d
//   r.sum_d_dt = L * s.sum_dt + s.sum_d_dt
//   r.sum_d2   = L^2 * s.n    + 2*L * s.sum_d + s.sum_d2
//
// combine(A, B): merge stats from disjoint tip sets --- element-wise addition.
//
// subtract(A, B): element-wise difference (inverse of combine).
//
// === Key relationship ===
//
// For a non-root node X with parent P (branch length L_X):
//
//   upstream_stats_at[X] = shift(all_stats_at[P] - shift(downstream_stats_at[X], L_X), L_X)
//
// i.e., start from all tips at P, subtract X's subtree contribution as seen from P,
// then move the vantage point from P to X (shifting all distances by L_X).
//
// === OLS at a candidate root position ===
//
// For a non-root node X with parent P (branch length L_X), consider placing a candidate
// root at fraction alpha in [0, 1] along the edge X<->P, at distance alpha*L_X from X
// and (1-alpha)*L_X from P.  (Edges are identified by non-root nodes, so the edge above
// the current tree root is never considered.)
//
//   root_stats(alpha) = shift(downstream_stats_at[X], alpha * L_X)
//                     + shift(upstream_stats_at[X], -alpha * L_X)
//
// Tips downstream of X get alpha*L_X farther from the candidate root; tips upstream of X
// get alpha*L_X closer.
//
// At alpha = 0, root_stats = all_stats_at[X].  At alpha = 1, root_stats = all_stats_at[P].
//
// The OLS quantities at position alpha:
//
//   mean_d    = root_stats.sum_d / N
//   Cov(d,t)  = root_stats.sum_d_dt / N       [since sum(dt_i) = 0]
//   Var(d)    = root_stats.sum_d2 / N  -  mean_d^2
//
//   R^2(alpha) = Cov(d,t)^2 / (Var(d) * Var(t))
//   RMS(alpha) = [N * Var(d) - N * Cov(d,t)^2 / Var(t)] / (N - 2)
//
// Note: Delphy filters candidates on Cov(d,t) > 0 (positive clock signal), but clockor2
// does not --- R^2 uses Cov(d,t)^2 (always non-negative) and RMS minimization is
// sign-agnostic.  We match clockor2's existing behavior here and do not filter.
//
// === Closed-form optimal alpha per edge ===
//
// All three sums are polynomial in alpha:
//
//   sum_d(alpha)    = s0 + s1 * alpha           (linear)
//   sum_d_dt(alpha) = p0 + p1 * alpha           (linear)
//   sum_d2(alpha)   = d0 + d1 * alpha + d2 * alpha^2  (quadratic)
//
// R^2 mode: R^2(alpha) is proportional to p(alpha)^2 / q(alpha) where p = sum_d_dt
// (linear), q = N * sum_d2 - sum_d^2 (quadratic).  Setting d/d(alpha)[p^2/q] = 0, the
// alpha^2 terms cancel (both the 2*p'*q and p*q' terms contribute 2*p1*q2*alpha^2),
// leaving:
//
//   alpha* = (2 * p1 * q0 - p0 * q1) / (2 * p0 * q2 - p1 * q1)
//
// RMS mode: RMS is proportional to q(alpha) - p(alpha)^2 / Var(t), which is a quadratic
// in alpha with minimum at alpha* = -b / (2*a).
//
// Clamp alpha* to [0, 1].  Evaluate the objective at alpha*, 0, and 1 (up to 3
// candidates per edge).  Total work per edge: O(1).
//
// === Algorithm ===
//
// Pass 1 (post-order): compute downstream_stats_at[X] for each node.
//   - Tip X: {n:1, sum_dt:dt_X, sum_d:0, sum_d_dt:0, sum_d2:0}
//   - Internal X with children C1, ..., Ck (branch lengths L1, ..., Lk):
//     downstream_stats_at[X] = shift(downstream_stats_at[C1], L1) + ... +
//                              shift(downstream_stats_at[Ck], Lk)
//
// Pass 2 (pre-order): compute upstream_stats_at[X] and all_stats_at[X].
//   - At the root R: all_stats_at[R] = downstream_stats_at[R] (there are no tips upstream of R).
//   - For each non-root node X with parent P (already visited in pre-order):
//     1. upstream_stats_at[X] = shift(all_stats_at[P] - shift(downstream_stats_at[X], L_X), L_X)
//     2. all_stats_at[X] = downstream_stats_at[X] + upstream_stats_at[X]
//
// Pass 3: for each non-root node X (identifying the edge X<->P), evaluate R^2/RMS
//   on edge X<->P using downstream_stats_at[X] and upstream_stats_at[X] via the
//   closed-form alpha*.  Track the overall best {node, alpha, value}.
//
// Total: O(N) time, O(N) space (three stats structs per node).

/**
 * Accumulated statistics for a set of tips, measured from a reference node.
 * Used to efficiently compute OLS regression quantities at candidate root positions.
 */
export interface RootingStats {
  n: number;        // number of tips
  sum_dt: number;   // sum of (t_i - mean_t), centered tip dates
  sum_d: number;    // sum of dist(ref, tip_i)
  sum_d_dt: number; // sum of dist(ref, tip_i) * (t_i - mean_t)
  sum_d2: number;   // sum of dist(ref, tip_i)^2
}

/**
 * Shift the reference point away by distance L, increasing all distances by L.
 */
export function shift(s: RootingStats, L: number): RootingStats {
  return {
    n: s.n,
    sum_dt: s.sum_dt,
    sum_d: L * s.n + s.sum_d,
    sum_d_dt: L * s.sum_dt + s.sum_d_dt,
    sum_d2: L * L * s.n + 2 * L * s.sum_d + s.sum_d2,
  };
}

/**
 * Merge stats from two disjoint tip sets --- element-wise addition.
 */
export function combine(a: RootingStats, b: RootingStats): RootingStats {
  return {
    n: a.n + b.n,
    sum_dt: a.sum_dt + b.sum_dt,
    sum_d: a.sum_d + b.sum_d,
    sum_d_dt: a.sum_d_dt + b.sum_d_dt,
    sum_d2: a.sum_d2 + b.sum_d2,
  };
}

/**
 * Element-wise difference of stats (inverse of combine).
 */
export function subtract(a: RootingStats, b: RootingStats): RootingStats {
  return {
    n: a.n - b.n,
    sum_dt: a.sum_dt - b.sum_dt,
    sum_d: a.sum_d - b.sum_d,
    sum_d_dt: a.sum_d_dt - b.sum_d_dt,
    sum_d2: a.sum_d2 - b.sum_d2,
  };
}

const ZERO_STATS: RootingStats = { n: 0, sum_dt: 0, sum_d: 0, sum_d_dt: 0, sum_d2: 0 };

/**
 * Best candidate root position found so far across all edges.
 */
interface EdgeCandidate {
  nodeIndex: number; // index into tree.nodeList identifying the edge X<->parent(X)
  alpha: number;     // fraction of the way from X to parent(X)
  value: number;     // objective to maximize: R^2, or -RMS (so best = max in both modes)
}

/**
 * Find the optimal root position along the edge connecting node X to its parent P.
 * The returned alpha is a fraction in [0, 1]: 0 means at X, 1 means at P.
 *
 * @param downstream - downstream_stats_at[X] for node X.
 * @param upstream - upstream_stats_at[X] for node X.
 * @param L - branch length of the edge X<->P.
 * @param N - total number of tips.
 * @param Var_t - variance of tip dates.
 * @param bfrMode - "R2" to maximize R^2, "RMS" to minimize residual mean square.
 * @returns the best alpha in [0, 1] and its objective value (always to maximize).
 */
function findBestAlphaOnEdge(
  downstream: RootingStats,
  upstream: RootingStats,
  L: number,
  N: number,
  Var_t: number,
  bfrMode: "R2" | "RMS",
): { alpha: number; value: number } {
  // Evaluate R^2 for a candidate root R' given all_stats_at[R']
  const evaluateR2 = (stats: RootingStats): number => {
    const mean_d = stats.sum_d / N;
    const Cov_dt = stats.sum_d_dt / N;
    const Var_d = stats.sum_d2 / N - mean_d * mean_d;
    if (Var_d <= 0) return 0;
    return (Cov_dt * Cov_dt) / (Var_d * Var_t);
  };

  // Evaluate -RMS for a candidate root R' given all_stats_at[R'].
  // Negated so that "best" is always the maximum across both bfrMode's.
  const evaluateNegRMS = (stats: RootingStats): number => {
    const mean_d = stats.sum_d / N;
    const Cov_dt = stats.sum_d_dt / N;
    const Var_d = stats.sum_d2 / N - mean_d * mean_d;
    return -(N * Var_d - N * Cov_dt * Cov_dt / Var_t) / (N - 2);
  };

  const evaluate = bfrMode === "R2" ? evaluateR2 : evaluateNegRMS;
  // root_stats(alpha) = shift(downstream, alpha*L) + shift(upstream, -alpha*L)
  //
  // sum_d(alpha)    = s0 + s1 * alpha
  // sum_d_dt(alpha) = p0 + p1 * alpha
  // sum_d2(alpha)   = d0 + d1 * alpha + d2 * alpha^2
  //
  // At alpha=0: root_stats = downstream + upstream = all_stats_at[X]
  // At alpha=1: root_stats = all_stats_at[P]

  const s0 = downstream.sum_d + upstream.sum_d;
  const s1 = L * downstream.n - L * upstream.n;

  const p0 = downstream.sum_d_dt + upstream.sum_d_dt;
  const p1 = L * downstream.sum_dt - L * upstream.sum_dt;

  const d0 = downstream.sum_d2 + upstream.sum_d2;
  const d1 = 2 * L * downstream.sum_d - 2 * L * upstream.sum_d;
  const d2 = L * L * downstream.n + L * L * upstream.n;

  let alphas: number[] = [0, 1];

  if (bfrMode === "R2") {
    // q(alpha) = N * sum_d2 - sum_d^2
    const q0 = N * d0 - s0 * s0;
    const q1 = N * d1 - 2 * s0 * s1;
    const q2 = N * d2 - s1 * s1;

    const denom = 2 * p0 * q2 - p1 * q1;
    if (Math.abs(denom) > 1e-30) {
      const alphaStar = (2 * p1 * q0 - p0 * q1) / denom;
      if (alphaStar > 0 && alphaStar < 1) {
        alphas.push(alphaStar);
      }
    }
  } else {
    // RMS proportional to q(alpha) - p(alpha)^2 / Var_t
    // = quadratic in alpha: a*alpha^2 + b*alpha + c
    // q coefficients
    const q0 = N * d0 - s0 * s0;
    const q1 = N * d1 - 2 * s0 * s1;
    const q2 = N * d2 - s1 * s1;

    const a = q2 - p1 * p1 / Var_t;
    const b = q1 - 2 * p0 * p1 / Var_t;

    if (Math.abs(a) > 1e-30) {
      const alphaStar = -b / (2 * a);
      if (alphaStar > 0 && alphaStar < 1) {
        alphas.push(alphaStar);
      }
    }
  }

  let bestAlpha = 0;
  let bestValue = -Infinity;

  for (const alpha of alphas) {
    const stats = combine(shift(downstream, alpha * L), shift(upstream, -alpha * L));
    const value = evaluate(stats);

    if (value > bestValue) {
      bestValue = value;
      bestAlpha = alpha;
    }
  }

  return { alpha: bestAlpha, value: bestValue };
}

/**
 * Convert alpha from the linear algorithm's convention to rerootAndScale's.
 *
 * The linear algorithm's alpha is relative to the edge length L (the branch
 * from X to its parent).  rerootAndScale expects alpha relative to the
 * post-reroot basal length (bl[0] + bl[1]).  These differ when X is a child
 * of the current root: rerooting collapses the old root, making the basal
 * length L + L_sibling rather than just L.
 *
 * Both this function and rerootAndScale assume that reroot(X) places X as
 * root.children[0].  This is the observed behavior of phylojs but is not
 * documented; if it changes, the basal branch split will be reversed.
 */
function alphaForReroot(tree: Tree, best: EdgeCandidate): number {
  const bestNode = tree.nodeList[best.nodeIndex];

  if (bestNode.parent !== tree.root) {
    return best.alpha;
  }

  const L = bestNode.branchLength ?? 0;
  let totalBasal = L;
  for (const child of tree.root.children) {
    if (child !== bestNode) {
      totalBasal += child.branchLength ?? 0;
    }
  }
  return (best.alpha * L) / totalBasal;
}

/**
 * Finds the best-fitting root for a phylogenetic tree in O(N) time.
 * Drop-in replacement for globalRootParallel with the same signature.
 *
 * @param nwk - Newick string representing the phylogenetic tree.
 * @param dates - array of dates associated with each tip (unused, dates come from tipData).
 * @param tipData - object mapping tip labels to { date, ... }.
 * @param bfrMode - "R2" to maximize R^2, "RMS" to minimize residual mean square.
 * @returns the rerooted tree as a Newick string.
 */
export async function globalRootLinear(
  nwk: string, dates: number[], tipData: any, bfrMode: "R2" | "RMS"
): Promise<string> {
  await new Promise(r => setTimeout(r, 0));

  const tree: Tree = readNewick(nwk);
  const nodes = tree.nodeList;
  const tips = tree.leafList;
  const N = tips.length;

  const startTime = new Date().getTime();

  // Regression needs at least 3 tips (2 parameters to fit)
  if (N <= 2) return nwk;

  // Assign node.id = index into nodes for O(1) lookup
  for (let i = 0; i < nodes.length; i++) {
    nodes[i].id = i;
  }

  // Build per-node date array indexed by node.id (assumes every tip has a unique label)
  const tipDate: number[] = new Array(nodes.length);
  for (const tip of tips) {
    tipDate[tip.id] = tipData[tip.label!].date;
  }

  // Pass 0: Global constants
  let sum_t = 0;
  for (const tip of tips) {
    sum_t += tipDate[tip.id];
  }
  const mean_t = sum_t / N;

  let sum_dt2 = 0;
  for (const tip of tips) {
    const dt = tipDate[tip.id] - mean_t;
    sum_dt2 += dt * dt;
  }
  const Var_t = sum_dt2 / N;

  // All dates equal --- regression slope is undefined, return input unchanged
  if (Var_t === 0) return nwk;

  // Allocate per-node stats arrays
  const downstreamStats: RootingStats[] = new Array(nodes.length);
  const upstreamStats: RootingStats[] = new Array(nodes.length);
  const allStats: RootingStats[] = new Array(nodes.length);

  // Pass 1: Bottom-up (post-order) --- compute downstream_stats_at[X]
  tree.applyPostOrder((node: any) => {
    if (node.isLeaf()) {
      const dt = tipDate[node.id] - mean_t;
      downstreamStats[node.id] = { n: 1, sum_dt: dt, sum_d: 0, sum_d_dt: 0, sum_d2: 0 };
    } else {
      let stats = ZERO_STATS;
      for (const child of node.children) {
        const L = child.branchLength ?? 0;
        stats = combine(stats, shift(downstreamStats[child.id], L));
      }
      downstreamStats[node.id] = stats;
    }
  });

  // Pass 2: Top-down (pre-order) --- compute upstream_stats_at[X] and all_stats_at[X]
  tree.applyPreOrder((node: any) => {
    if (node.isRoot()) {
      upstreamStats[node.id] = ZERO_STATS;
      allStats[node.id] = downstreamStats[node.id];
    } else {
      const L = node.branchLength ?? 0;
      upstreamStats[node.id] = shift(
        subtract(allStats[node.parent.id], shift(downstreamStats[node.id], L)),
        L
      );
      allStats[node.id] = combine(downstreamStats[node.id], upstreamStats[node.id]);
    }
  });

  // Pass 3: Best candidate per edge (always maximize --- RMS is negated)
  let best: EdgeCandidate = {
    nodeIndex: 0,
    alpha: 0.5,
    value: -Infinity,
  };

  for (let i = 0; i < nodes.length; i++) {
    if (nodes[i].isRoot()) continue;
    const L = nodes[i].branchLength ?? 0;
    const result = findBestAlphaOnEdge(
      downstreamStats[i], upstreamStats[i], L, N, Var_t, bfrMode
    );
    if (result.value > best.value) {
      best = { nodeIndex: i, alpha: result.alpha, value: result.value };
    }
  }

  // Reroot and adjust basal branch lengths
  console.log("Overall Best");
  console.log(best);
  const bestTree = readNewick(nwk);
  const alpha = alphaForReroot(bestTree, best);
  rerootAndScale(bestTree, { nodeIndx: best.nodeIndex, alpha });

  const endTime = new Date().getTime();
  console.log("Time taken for BFR (linear) " + (endTime - startTime) + "ms");

  bestTree.ladderise();
  return writeNewick(bestTree);
}
