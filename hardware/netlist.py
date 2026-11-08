"""
Read EvoApproxLib's gate-level Verilog for an 8x8 multiplier, simulate it, and write it as BLIF.

Each netlist is a list of `assign` statements built from &, |, ^, ~ and constants. We use it to
  * check that the netlist computes exactly the lookup table we use in the experiments,
  * count gates and simulate how many gate outputs switch for a given stream of inputs
    (switching activity is what drives dynamic power), and
  * export BLIF so the logic synthesis tool ABC can optimize and map it (see synth.sh).
"""
import glob, itertools, os, re
import numpy as np

ASSIGN = re.compile(r"\s*assign\s+(\S+)\s*=\s*(.+?)\s*;")
NAME = re.compile(r"[A-Za-z_]\w*(?:\[\d+\])?")
CELL = re.compile(r"\s*(PDK\w+)\s+(?:(\w+))\s*\((.*)\)\s*;")
PIN = re.compile(r"\.(\w+)\(([^)]*)\)")


def find_netlist(lib, name):
    hits = [f for f in glob.glob(os.path.join(lib, "multiplers", "*", "*", name + ".v"))]
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


class Netlist:
    def __init__(self, path):
        self.name = os.path.basename(path)[:-2]
        self.assigns = []                                   # (target, python expression, input names)
        for line in open(path):
            if line.strip().startswith("endmodule"):      # later modules define the library cells
                break
            m = ASSIGN.match(line)
            if m:
                lines = [(m.group(1), m.group(2))]
            else:
                c = CELL.match(line)                        # the exact multiplier uses library cells
                if not c:
                    continue
                pins = dict(PIN.findall(c.group(3)))
                kind, t = c.group(1), f"t_{c.group(2)}"
                if kind == "PDKGENAND2X1":
                    lines = [(pins["Y"], f"{pins['A']} & {pins['B']}")]
                elif kind == "PDKGENHAX1":
                    lines = [(pins["YS"], f"{pins['A']} ^ {pins['B']}"), (pins["YC"], f"{pins['A']} & {pins['B']}")]
                elif kind == "PDKGENFAX1":                  # full adder as 2 XOR, 2 AND, 1 OR
                    lines = [(t + "x", f"{pins['A']} ^ {pins['B']}"), (pins["YS"], f"{t}x ^ {pins['C']}"),
                             (t + "a", f"{pins['A']} & {pins['B']}"), (t + "b", f"{t}x & {pins['C']}"),
                             (pins["YC"], f"{t}a | {t}b")]
                else:
                    raise ValueError(f"unknown cell {kind} in {path}")
            for lhs, rhs in lines:
                self._add(lhs, rhs)
        self.gates = [a for a in self.assigns if a[2] and not re.fullmatch(r"[A-Za-z_]\w*(\[\d+\])?", a[1].strip())]

    def _add(self, lhs, rhs):
        rhs = rhs.replace("1'b0", "ZERO").replace("1'b1", "ONE")
        names = [n for n in NAME.findall(rhs) if n not in ("ZERO", "ONE")]
        self.assigns.append((lhs, rhs, names))

    @staticmethod
    def _key(n):
        return n.replace("[", "_").replace("]", "")

    def simulate(self, a, b):
        """a, b: uint8 arrays of equal length -> (product array, dict of every signal's bool array)."""
        env = {f"A_{i}": ((a >> i) & 1).astype(bool) for i in range(8)}
        env.update({f"B_{i}": ((b >> i) & 1).astype(bool) for i in range(8)})
        env["ZERO"] = np.zeros(len(a), bool); env["ONE"] = np.ones(len(a), bool)
        for lhs, rhs, names in self.assigns:
            expr = NAME.sub(lambda m: self._key(m.group(0)), rhs)
            env[self._key(lhs)] = np.asarray(eval(expr, {}, env), bool) if names or "ZERO" in rhs or "ONE" in rhs \
                else env[self._key(rhs)]
        out = sum(env.get(f"O_{i}", env["ZERO"]).astype(np.int64) << i for i in range(16))
        return out, env

    def gate_counts(self):
        """Gates as written in the library netlist, by operator."""
        c = {"AND": 0, "OR": 0, "XOR": 0, "NOT": 0, "complex": 0}
        for _, rhs, _ in self.gates:
            ops = re.findall(r"[&|^~]", rhs)
            if len(ops) == 1:
                c[{"&": "AND", "|": "OR", "^": "XOR", "~": "NOT"}[ops[0]]] += 1
            else:
                c["complex"] += 1
        return c

    def switching(self, a, b):
        """Average number of gate outputs that change value from one input pair to the next."""
        _, env = self.simulate(a, b)
        tog = 0.0
        for lhs, _, _ in self.gates:
            s = env[self._key(lhs)]
            tog += np.count_nonzero(s[1:] != s[:-1])
        return tog / (len(a) - 1)

    def to_blif(self, path):
        """Each assign becomes one .names block, its on-set listed row by row."""
        lines = [f".model {self.name}",
                 ".inputs " + " ".join([f"A_{i}" for i in range(8)] + [f"B_{i}" for i in range(8)]),
                 ".outputs " + " ".join(f"O_{i}" for i in range(16))]
        defined = set()
        for lhs, rhs, names in self.assigns:
            tgt = self._key(lhs)
            ins = list(dict.fromkeys(self._key(n) for n in names))
            if not ins:                                      # constant
                lines += [f".names {tgt}"] + (["1"] if "ONE" in rhs else [])
            else:
                expr = NAME.sub(lambda m: self._key(m.group(0)), rhs)
                lines.append(".names " + " ".join(ins + [tgt]))
                combos = np.array(list(itertools.product((0, 1), repeat=len(ins))), bool)   # every input row
                env = {n: combos[:, k] for k, n in enumerate(ins)}
                env["ZERO"] = np.zeros(len(combos), bool); env["ONE"] = np.ones(len(combos), bool)
                val = np.asarray(eval(expr, {}, env), bool) if len(ins) > 1 or expr.strip() != ins[0] \
                    else combos[:, 0]
                for bits, v in zip(combos, val):
                    if v:
                        lines.append("".join("1" if x else "0" for x in bits) + " 1")
            defined.add(tgt)
        for i in range(16):                                  # outputs never assigned are 0
            if f"O_{i}" not in defined:
                lines.append(f".names O_{i}")
        lines.append(".end")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
