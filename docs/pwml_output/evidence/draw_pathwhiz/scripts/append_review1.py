"""Fill the A/B paragraph of review1.md.txt and append it to DRAW-PATHWHIZ.md once."""
AB = """**PathWhiz A/B on `23ffd20c`: NOT RUN (blocked by memory, not by a failure).** The re-gate was first one hold
(`final3.sh`); with Docker up the machine had 3 GB free, under the queue's 4 GB floor, so the gate never opened.
I stopped Docker Desktop (no container was running; I had started it), split the A/B into `final4.sh`, and armed
`start_docker_then_queue.sh` (waits for an idle lock and >= 7 GB free, starts Docker outside the bounded wrapper,
then queues `final4.sh`). Starting Docker once at 4 GB free took the machine to 0-1 GB free, so I stopped it again.
Free RAM then stayed at 4 GB for 40 minutes with the lock idle (41 GB committed on 31 GB physical), so the waiter was
stopped. The stack `t2pw-pwdraw` is down, volumes kept, Docker Desktop stopped.

What the earlier A/B (`4b701677`, section 3) still covers: F1 changes geometry only (content check 0), and only in
lipid A, chlorophyll, sour taste and V7b idx 7, 10, 126 of the A/B set; the other 8 files are byte-identical to the
ones imported then (created counts equal, messages unchanged). Not covered yet: those 6 files on `23ffd20c`, the F3
files (idx 11, 134, 114) and the missing-template probe.

To run it when memory allows (one command, it waits and queues itself):
`"C:/Program Files/Git/bin/bash.exe" C:/t/lane-pwdraw-work/scripts/start_docker_then_queue.sh`
then `pw_ab.py pw/base2 pw/after5 OUT.json` and `compare_page.py` (point it at `after5`)."""
W = "C:/t/lane-pwdraw-work/scripts/"
s = open(W + "review1.md.txt", encoding="utf-8").read().replace("__AB__", AB)
t = "C:/t/lane-pwdraw/docs/pwml_output/DRAW-PATHWHIZ.md"
cur = open(t, encoding="utf-8").read()
assert "## 6. Review round 1" not in cur
open(t, "a", encoding="utf-8", newline="\n").write(s)
print("appended")
