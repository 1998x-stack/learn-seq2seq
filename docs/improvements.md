# Portfolio Engineering Improvements

### 2026-11-05 12:20 UTC — ChatGPT Portfolio Iteration
- Run ID: 20261105T1220Z-9
- Repo / Base SHA: 1998x-stack/learn-seq2seq / 821cc0b81152d195579fac2a6e21f65da2b5d8ab
- Historical context: Default branch docs/improvements.md returned 404; first initialization. GitHub PR search found no prior PRs.
- Product problem & expected user value: m03 tutorial incorrectly teaches that ignoring padding must reduce cross-entropy; learners can encounter random assertion failures and misunderstand reduction semantics.
- Engineering diagnosis & evidence: m03_batching/code.py main block compares random full-batch loss and masked loss with strict less-than; masking can increase or decrease mean loss depending on removed examples. A separate educational risk is that the random sample does not guarantee direction.
- Options & decision: Fix m03 invalid loss assertion (priority 14); revisit beam-search EOS handling (deferred); investigate training/evaluation repeatability (deferred). Selected local and backwards-compatible test correction.
- Changes: m03_batching/code.py now verifies retained-target equivalence instead of a false inequality; tests/test_m03_padding.py covers masking equality and both directional counterexamples. Public module API unchanged.
- Validation: A: local Python 3 / torch 2.10.0+cpu independently recreated three test cases and all passed (0.134s); D: repository's exact committed test file and full m03 script were not executed in a checkout because GitHub clone network access was unavailable; C: code and diff inspected statically.
- Security & regression review: No credentials, network calls, permissions, CI workflows, deployment, or production data changed. Limited to tutorial and regression test.
- Risk & rollback: Low risk; revert the feature branch commits or close the Draft PR. Full script smoke test and CI review remain pending.
- Follow-up: Run python m03_batching/code.py and python -m unittest discover -s tests -p 'test_m03_padding.py' on a real checkout.
- Commit: pending
- PR: pending
