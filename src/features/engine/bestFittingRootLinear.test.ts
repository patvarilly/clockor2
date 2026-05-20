import {
  globalRootLinear,
  shift,
  combine,
  subtract,
  RootingStats,
} from './bestFittingRootLinear';
import { localRootR2, localRootRMS, rerootAndScale } from './bestFittingRoot';
import { Tree, readNewick, writeNewick } from 'phylojs';
import { decimal_date, extractPartOfTipName } from './utils';
import { readFileSync } from 'fs';

// === Helpers ===

function makeTipDataFromNwk(nwk: string): any {
  const tree = readNewick(nwk);
  const tipNames = tree.getTipLabels();
  const tipData: any = {};
  for (const name of tipNames) {
    const dateStr = extractPartOfTipName(name, "_", "-1");
    tipData[name] = { date: decimal_date(dateStr, "yyyy-mm-dd") };
  }
  return tipData;
}

// Brute-force BFR by iterating over all edges (same approach as
// the MOCK globalRootParallel test in bestFittingRoot.test.ts).
// Returns the rerooted, ladderised Newick string and the best objective value.
function bruteForceR2(nwk: string, tipData: any): { nwk: string; bestR2: number } {
  const tree = readNewick(nwk);
  const nodes = tree.nodeList;
  let bestR2 = -Infinity;
  let bestNodeIndex = 0;
  let bestAlpha = 0.5;

  for (let i = 0; i < nodes.length; i++) {
    const tr = readNewick(nwk);
    const n = tr.nodeList[i];
    if (!n.isRoot()) tr.reroot(n);
    const result = localRootR2(tr, tipData);
    if (result.value > bestR2) {
      bestR2 = result.value;
      bestNodeIndex = i;
      bestAlpha = result.alpha;
    }
  }

  const bestTree = readNewick(nwk);
  rerootAndScale(bestTree, { nodeIndx: bestNodeIndex, alpha: bestAlpha });
  bestTree.ladderise();
  return { nwk: writeNewick(bestTree), bestR2 };
}

function bruteForceRMS(nwk: string, tipData: any): { nwk: string; bestRMS: number } {
  const tree = readNewick(nwk);
  const nodes = tree.nodeList;
  let bestRMS = Infinity;
  let bestNodeIndex = 0;
  let bestAlpha = 0.5;

  for (let i = 0; i < nodes.length; i++) {
    const tr = readNewick(nwk);
    const n = tr.nodeList[i];
    if (!n.isRoot()) tr.reroot(n);
    const result = localRootRMS(tr, tipData);
    if (result.value < bestRMS) {
      bestRMS = result.value;
      bestNodeIndex = i;
      bestAlpha = result.alpha;
    }
  }

  const bestTree = readNewick(nwk);
  rerootAndScale(bestTree, { nodeIndx: bestNodeIndex, alpha: bestAlpha });
  bestTree.ladderise();
  return { nwk: writeNewick(bestTree), bestRMS };
}

// === RootingStats operations ===

describe('RootingStats operations', () => {
  test('shift adds distance L to all tips', () => {
    const s: RootingStats = { n: 1, sum_dt: 0, sum_d: 0, sum_d_dt: 0, sum_d2: 0 };
    const r = shift(s, 5);
    expect(r.n).toBe(1);
    expect(r.sum_dt).toBe(0);
    expect(r.sum_d).toBe(5);
    expect(r.sum_d_dt).toBe(0);
    expect(r.sum_d2).toBe(25);
  });

  test('shift with nonzero dt', () => {
    const s: RootingStats = { n: 2, sum_dt: 3, sum_d: 4, sum_d_dt: 5, sum_d2: 6 };
    const r = shift(s, 10);
    expect(r.n).toBe(2);
    expect(r.sum_dt).toBe(3);
    expect(r.sum_d).toBe(10 * 2 + 4);      // 24
    expect(r.sum_d_dt).toBe(10 * 3 + 5);    // 35
    expect(r.sum_d2).toBe(100 * 2 + 2 * 10 * 4 + 6);  // 286
  });

  test('combine is element-wise addition', () => {
    const a: RootingStats = { n: 1, sum_dt: 2, sum_d: 3, sum_d_dt: 4, sum_d2: 5 };
    const b: RootingStats = { n: 10, sum_dt: 20, sum_d: 30, sum_d_dt: 40, sum_d2: 50 };
    const c = combine(a, b);
    expect(c.n).toBe(11);
    expect(c.sum_dt).toBe(22);
    expect(c.sum_d).toBe(33);
    expect(c.sum_d_dt).toBe(44);
    expect(c.sum_d2).toBe(55);
  });

  test('subtract is element-wise difference', () => {
    const a: RootingStats = { n: 11, sum_dt: 22, sum_d: 33, sum_d_dt: 44, sum_d2: 55 };
    const b: RootingStats = { n: 1, sum_dt: 2, sum_d: 3, sum_d_dt: 4, sum_d2: 5 };
    const c = subtract(a, b);
    expect(c.n).toBe(10);
    expect(c.sum_dt).toBe(20);
    expect(c.sum_d).toBe(30);
    expect(c.sum_d_dt).toBe(40);
    expect(c.sum_d2).toBe(50);
  });

  test('combine(a, b) then subtract(result, b) recovers a', () => {
    const a: RootingStats = { n: 3, sum_dt: -1.5, sum_d: 7.2, sum_d_dt: -3.1, sum_d2: 12.4 };
    const b: RootingStats = { n: 2, sum_dt: 1.5, sum_d: 4.8, sum_d_dt: 2.1, sum_d2: 8.6 };
    const c = subtract(combine(a, b), b);
    expect(c.n).toBeCloseTo(a.n);
    expect(c.sum_dt).toBeCloseTo(a.sum_dt);
    expect(c.sum_d).toBeCloseTo(a.sum_d);
    expect(c.sum_d_dt).toBeCloseTo(a.sum_d_dt);
    expect(c.sum_d2).toBeCloseTo(a.sum_d2);
  });

  test('shift by 0 is identity', () => {
    const s: RootingStats = { n: 3, sum_dt: 1, sum_d: 2, sum_d_dt: 3, sum_d2: 4 };
    const r = shift(s, 0);
    expect(r).toEqual(s);
  });

  test('hand-computed 3-tip example', () => {
    // Tree: ((A:2, B:3):0, C:5):0  with dates A=0, B=4, C=2
    // mean_t = 2, so dt_A = -2, dt_B = 2, dt_C = 0
    //
    // downstream_stats_at[A] = {n:1, sum_dt:-2, sum_d:0, sum_d_dt:0, sum_d2:0}
    // downstream_stats_at[B] = {n:1, sum_dt:2, sum_d:0, sum_d_dt:0, sum_d2:0}
    // downstream_stats_at[AB] = shift({1,-2,0,0,0}, 2) + shift({1,2,0,0,0}, 3)
    //   shift(A, 2) = {1, -2, 2, -4, 4}
    //   shift(B, 3) = {1, 2, 3, 6, 9}
    //   combined   = {2, 0, 5, 2, 13}

    const statsA: RootingStats = { n: 1, sum_dt: -2, sum_d: 0, sum_d_dt: 0, sum_d2: 0 };
    const statsB: RootingStats = { n: 1, sum_dt: 2, sum_d: 0, sum_d_dt: 0, sum_d2: 0 };

    const shiftedA = shift(statsA, 2);
    expect(shiftedA).toEqual({ n: 1, sum_dt: -2, sum_d: 2, sum_d_dt: -4, sum_d2: 4 });

    const shiftedB = shift(statsB, 3);
    expect(shiftedB).toEqual({ n: 1, sum_dt: 2, sum_d: 3, sum_d_dt: 6, sum_d2: 9 });

    const statsAB = combine(shiftedA, shiftedB);
    expect(statsAB).toEqual({ n: 2, sum_dt: 0, sum_d: 5, sum_d_dt: 2, sum_d2: 13 });
  });
});

// === Agreement between globalRootLinear and brute-force ===

function getRTTByTip(tree: Tree): Map<string, number> {
  const dists = tree.getRTTDist();
  const tips = tree.getTipLabels();
  const map = new Map<string, number>();
  for (let i = 0; i < tips.length; i++) {
    map.set(tips[i], dists[i]);
  }
  return map;
}

function expectTreesEqual(linearTree: Tree, bruteTree: Tree): void {
  const linearRTT = getRTTByTip(linearTree);
  const bruteRTT = getRTTByTip(bruteTree);

  expect(linearRTT.size).toBe(bruteRTT.size);
  linearRTT.forEach((dist, tip) => {
    expect(bruteRTT.has(tip)).toBe(true);
    expect(dist).toBeCloseTo(bruteRTT.get(tip)!, 6);
  });

  expect(linearTree.getTotalBranchLength()).toBeCloseTo(
    bruteTree.getTotalBranchLength(), 6
  );
}

describe('globalRootLinear agrees with brute-force', () => {
  test('4-tip tree R2', async () => {
    const nwk = '((A:1, B:2):3,(C:1, D:2):1);';
    const tipData = {
      A: { date: 2 },
      B: { date: 3 },
      C: { date: 4 },
      D: { date: 5 },
    };

    const linearNwk = await globalRootLinear(nwk, [], tipData, "R2");
    const bruteNwk = bruteForceR2(nwk, tipData).nwk;
    expectTreesEqual(readNewick(linearNwk), readNewick(bruteNwk));
  });

  test('4-tip tree RMS', async () => {
    const nwk = '((A:1, B:2):3,(C:1, D:2):1);';
    const tipData = {
      A: { date: 2 },
      B: { date: 3 },
      C: { date: 4 },
      D: { date: 5 },
    };

    const linearNwk = await globalRootLinear(nwk, [], tipData, "RMS");
    const bruteNwk = bruteForceRMS(nwk, tipData).nwk;
    expectTreesEqual(readNewick(linearNwk), readNewick(bruteNwk));
  });

  test('3-tip tree R2', async () => {
    const nwk = '((A_2001:0.5,B_2009:1):1,C_2002:1.75):0;';
    const tipData = makeTipDataFromNwk(nwk);

    const linearNwk = await globalRootLinear(nwk, [], tipData, "R2");
    const bruteNwk = bruteForceR2(nwk, tipData).nwk;
    expectTreesEqual(readNewick(linearNwk), readNewick(bruteNwk));
  });

  test('3-tip tree RMS', async () => {
    const nwk = '((A_2001:0.5,B_2009:1):1,C_2002:1.75):0;';
    const tipData = makeTipDataFromNwk(nwk);

    const linearNwk = await globalRootLinear(nwk, [], tipData, "RMS");
    const bruteNwk = bruteForceRMS(nwk, tipData).nwk;
    expectTreesEqual(readNewick(linearNwk), readNewick(bruteNwk));
  });

  test('empirical tree R2', async () => {
    const nwk = readFileSync("src/features/engine/empiricalTestTree.nwk").toString();
    const tipData = makeTipDataFromNwk(nwk);

    const brute = bruteForceR2(nwk, tipData);
    expect(brute.bestR2).toBeCloseTo(0.173, 3);

    const linearNwk = await globalRootLinear(nwk, [], tipData, "R2");
    expectTreesEqual(readNewick(linearNwk), readNewick(brute.nwk));
  });

  test('empirical tree RMS', async () => {
    const nwk = readFileSync("src/features/engine/empiricalTestTree.nwk").toString();
    const tipData = makeTipDataFromNwk(nwk);

    const linearNwk = await globalRootLinear(nwk, [], tipData, "RMS");
    const bruteNwk = bruteForceRMS(nwk, tipData).nwk;
    expectTreesEqual(readNewick(linearNwk), readNewick(bruteNwk));
  });
});

// === Edge cases ===

describe('globalRootLinear edge cases', () => {
  test('all-same-date tips returns input unchanged', async () => {
    const nwk = '((A:1, B:2):3,(C:1, D:2):1);';
    const tipData = {
      A: { date: 5 },
      B: { date: 5 },
      C: { date: 5 },
      D: { date: 5 },
    };

    const result = await globalRootLinear(nwk, [], tipData, "R2");
    expect(result).toBe(nwk);
  });

  test('2-tip tree returns input unchanged', async () => {
    const nwk = '(A:1, B:2);';
    const tipData = {
      A: { date: 1 },
      B: { date: 2 },
    };

    const result = await globalRootLinear(nwk, [], tipData, "R2");
    expect(result).toBe(nwk);
  });

  test('star tree (all tips directly connected to root)', async () => {
    const nwk = '(A:1, B:2, C:3);';
    const tipData = {
      A: { date: 1 },
      B: { date: 2 },
      C: { date: 3 },
    };

    const resultNwk = await globalRootLinear(nwk, [], tipData, "R2");
    const resultTree = readNewick(resultNwk);
    const origTree = readNewick(nwk);

    expect(resultTree.getTotalBranchLength()).toBeCloseTo(origTree.getTotalBranchLength(), 6);
  });

  test('zero-length branches', async () => {
    const nwk = '((A:0, B:1):0,(C:1, D:2):0);';
    const tipData = {
      A: { date: 1 },
      B: { date: 2 },
      C: { date: 3 },
      D: { date: 4 },
    };

    const resultNwk = await globalRootLinear(nwk, [], tipData, "R2");
    const resultTree = readNewick(resultNwk);
    const origTree = readNewick(nwk);

    expect(resultTree.getTotalBranchLength()).toBeCloseTo(origTree.getTotalBranchLength(), 6);
  });
});
