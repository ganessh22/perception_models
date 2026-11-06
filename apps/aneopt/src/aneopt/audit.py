"""Static checks on a CoreML package: which activations pass the Neural Engine's on-chip working-set limit.

It reads the package's program and needs no hardware, so it can run before any benchmark.
"""
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# A single activation larger than this streams through memory. Measured with `aneopt sweep` or reported in published work:
#   M1: about 2 MB (arXiv 2606.22283).  M5: 4.72 MB (same paper).
#   M2 Pro: the throughput step falls between 1792 and 2048 KiB (our sweep).  Others: unknown, so run the sweep.
# The default is the largest size known to be safe on an M2 Pro.
DEFAULT_LIMIT_BYTES = 1792 * 1024

BYTES = {"FLOAT16": 2, "FLOAT32": 4, "FLOAT64": 8, "INT8": 1, "UINT8": 1, "INT16": 2, "UINT16": 2, "INT32": 4, "INT64": 8, "BOOL": 1}
SKIP = ("const",)  # weights and constants, reported separately


@dataclass
class Activation:
    op: str
    op_type: str
    shape: tuple
    bytes: int

    @property
    def padded_bytes(self):
        """The last axis of a Neural Engine buffer is aligned to 64 bytes (Apple's published guidance)."""
        size = self.bytes // max(1, _prod(self.shape))
        last = self.shape[-1] * size if self.shape else size
        return _prod(self.shape[:-1]) * ((last + 63) // 64 * 64)


@dataclass
class Audit:
    activations: list = field(default_factory=list)
    weight_bytes: int = 0
    op_counts: Counter = field(default_factory=Counter)

    def over(self, limit=DEFAULT_LIMIT_BYTES):
        return sorted((a for a in self.activations if a.padded_bytes > limit), key=lambda a: -a.padded_bytes)


def _prod(xs):
    out = 1
    for x in xs:
        out *= x
    return out


def _shape_and_bytes(tensor_type):
    from coremltools.proto import MIL_pb2

    dims = tensor_type.dimensions
    if any(d.WhichOneof("dimension") != "constant" for d in dims):
        return None, 0
    shape = tuple(d.constant.size for d in dims)
    name = MIL_pb2.DataType.Name(tensor_type.dataType)
    return shape, _prod(shape) * BYTES.get(name, 2)


def audit_package(path):
    """Read every op's output shape and every weight's size from an .mlpackage."""
    import coremltools as ct

    spec = ct.models.MLModel(str(Path(path)), skip_model_load=True).get_spec()
    if spec.WhichOneof("Type") != "mlProgram":
        raise ValueError("aneopt audit only reads ML programs (convert_to='mlprogram')")
    result = Audit()
    for function in spec.mlProgram.functions.values():
        for block in function.block_specializations.values():
            for op in block.operations:
                is_const = op.type in SKIP or op.type.startswith("constexpr")
                if not is_const:
                    result.op_counts[op.type] += 1
                for out in op.outputs:
                    if out.type.WhichOneof("type") != "tensorType":
                        continue
                    shape, size = _shape_and_bytes(out.type.tensorType)
                    if shape is None:
                        continue
                    if is_const:
                        if op.type != "const" or size > 1024:
                            result.weight_bytes += size
                    else:
                        result.activations.append(Activation(out.name, op.type, shape, size))
    return result


def print_audit(path, limit=DEFAULT_LIMIT_BYTES, top=20):
    a = audit_package(path)
    over = a.over(limit)
    print(f"{Path(path).name}: {sum(a.op_counts.values())} ops, {len(a.activations)} activations")
    if a.weight_bytes:
        print(f"weights and large constants, as stored: {a.weight_bytes / 1e6:.0f} MB. Compressed weights count once per stored array.")
    print(f"activations over {limit / 1024:.0f} KiB: {len(over)}")
    by_kind = Counter((x.op_type, x.shape, x.padded_bytes) for x in over)
    for (op_type, shape, size), n in by_kind.most_common(top):
        print(f"  {size / 1e6:6.2f} MB  x{n:<4d} {op_type:18s} {list(shape)}")
    if not over:
        print("  none: this graph shouldn't need any activation streamed through memory.")
    return a
