"""Read-only sampler for macOS IOReport counters: memory traffic per SoC block and power-state residency.

IOReport is a private Apple API, so channel names can change between chips and macOS versions. It needs no sudo.
On macOS 27 on an M2 Pro the "Energy Model" channels for the Neural Engine read zero, so use `sudo powermetrics` for watts.
"""
import ctypes as C

_cf = C.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
_io = C.CDLL("/usr/lib/libIOReport.dylib")
V = C.c_void_p
UTF8 = 0x08000100

_cf.CFStringCreateWithCString.restype = V
_cf.CFStringCreateWithCString.argtypes = [V, C.c_char_p, C.c_uint32]
_cf.CFDictionaryGetValue.restype = V
_cf.CFDictionaryGetValue.argtypes = [V, V]
_cf.CFArrayGetCount.restype = C.c_long
_cf.CFArrayGetCount.argtypes = [V]
_cf.CFArrayGetValueAtIndex.restype = V
_cf.CFArrayGetValueAtIndex.argtypes = [V, C.c_long]
_cf.CFStringGetCString.argtypes = [V, C.c_char_p, C.c_long, C.c_uint32]
_cf.CFRelease.argtypes = [V]

_io.IOReportCopyChannelsInGroup.restype = V
_io.IOReportCopyChannelsInGroup.argtypes = [V, V, C.c_uint64, C.c_uint64, C.c_uint64]
_io.IOReportMergeChannels.argtypes = [V, V, V]
_io.IOReportCreateSubscription.restype = V
_io.IOReportCreateSubscription.argtypes = [V, V, C.POINTER(V), C.c_uint64, V]
_io.IOReportCreateSamples.restype = V
_io.IOReportCreateSamples.argtypes = [V, V, V]
_io.IOReportCreateSamplesDelta.restype = V
_io.IOReportCreateSamplesDelta.argtypes = [V, V, V]
for name in ("IOReportChannelGetGroup", "IOReportChannelGetSubGroup", "IOReportChannelGetChannelName"):
    getattr(_io, name).restype = V
    getattr(_io, name).argtypes = [V]
_io.IOReportChannelGetFormat.restype = C.c_int
_io.IOReportChannelGetFormat.argtypes = [V]
_io.IOReportSimpleGetIntegerValue.restype = C.c_int64
_io.IOReportSimpleGetIntegerValue.argtypes = [V, C.c_int32]
_io.IOReportStateGetCount.restype = C.c_int32
_io.IOReportStateGetCount.argtypes = [V]
_io.IOReportStateGetNameForIndex.restype = V
_io.IOReportStateGetNameForIndex.argtypes = [V, C.c_int32]
_io.IOReportStateGetResidency.restype = C.c_int64
_io.IOReportStateGetResidency.argtypes = [V, C.c_int32]


def _cfs(text):
    return _cf.CFStringCreateWithCString(None, text.encode(), UTF8)


def _str(ref):
    if not ref:
        return ""
    buf = C.create_string_buffer(512)
    _cf.CFStringGetCString(ref, buf, 512, UTF8)
    return buf.value.decode()


class Sampler:
    def __init__(self, groups=(("AMC Stats", "Perf Counters"), ("Energy Model", None), ("SoC Stats", "Cluster Power States"), ("PMP", "DCS Floor"))):
        channels = None
        for group, sub in groups:
            found = _io.IOReportCopyChannelsInGroup(_cfs(group), _cfs(sub) if sub else None, 0, 0, 0)
            if channels is None:
                channels = found
            else:
                _io.IOReportMergeChannels(channels, found, None)
        self.subscribed = V()
        self.sub = _io.IOReportCreateSubscription(None, channels, C.byref(self.subscribed), 0, None)
        if not self.sub:
            raise RuntimeError("IOReport subscription failed")

    def snapshot(self):
        return _io.IOReportCreateSamples(self.sub, self.subscribed, None)

    def delta(self, start, end):
        """{(group, subgroup, channel): int, or {state: residency} for state channels}."""
        d = _io.IOReportCreateSamplesDelta(start, end, None)
        arr = _cf.CFDictionaryGetValue(d, _cfs("IOReportChannels"))
        out = {}
        for i in range(_cf.CFArrayGetCount(arr)):
            c = _cf.CFArrayGetValueAtIndex(arr, i)
            key = tuple(_str(f(c)) for f in (_io.IOReportChannelGetGroup, _io.IOReportChannelGetSubGroup, _io.IOReportChannelGetChannelName))
            fmt = _io.IOReportChannelGetFormat(c)
            if fmt == 1:
                out[key] = _io.IOReportSimpleGetIntegerValue(c, 0)
            elif fmt == 2:
                out[key] = {_str(_io.IOReportStateGetNameForIndex(c, s)): _io.IOReportStateGetResidency(c, s) for s in range(_io.IOReportStateGetCount(c))}
        return out
