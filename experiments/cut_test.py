"""Exercise the mid-run ensemble cut: force the planner to pick 8 members on a
budget that cannot afford them and check that coverage is kept.

    SIMULACRA_BUDGET=150 SIMULACRA_FORCE_PHASE=final python cut_test.py SUBDIR DATA_DIR [KIT]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluate as ev  # noqa: E402


def main():
    sub, data = sys.argv[1], sys.argv[2]
    kit = sys.argv[3] if len(sys.argv) > 3 else ev.DEFAULT_KIT
    m = ev.load_submission(sub)
    m._pick_nest = lambda *a, **k: 8
    prep = ev.prepare(data, 2, kit)
    vectors = m.predict(prep["frame"].copy(), prep["schema"])
    result = ev.grade(prep, vectors)
    info = m.LAST_INFO
    print("cut test skill %.5f" % result["skill"],
          {k: info.get(k) for k in ("budget_s", "nest_pfn", "nest_pfn_cut", "cover_pfn",
                                    "targets", "timings")}, flush=True)


if __name__ == "__main__":
    main()
