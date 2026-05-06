# RecordBatchFileWriter.close() and File Descriptor Lifecycle

This document traces what happens when `RecordBatchFileWriter.close()` is called
and when the underlying file descriptor is actually closed.

## Python Layer

`RecordBatchFileWriter` (in `python/pyarrow/ipc.py`) inherits from
`_RecordBatchFileWriter` (Cython, `python/pyarrow/ipc.pxi:1106`), which inherits
from `_RecordBatchStreamWriter`, which inherits from `_CRecordBatchWriter`.

The `close()` method lives on `_CRecordBatchWriter` (`ipc.pxi:619`):

```python
def close(self):
    with nogil:
        check_status(self.writer.get().Close())
```

This calls straight into the C++ `RecordBatchWriter::Close()`.

## C++ Layer

### IpcFormatWriter::Close() (`cpp/src/arrow/ipc/writer.cc:1246`)

`MakeFileWriter()` constructs an `IpcFormatWriter` wrapping a `PayloadFileWriter`.
`IpcFormatWriter::Close()` delegates to the payload writer:

```cpp
Status Close() override {
    RETURN_NOT_OK(CheckStarted());
    RETURN_NOT_OK(payload_writer_->Close());
    closed_ = true;
    return Status::OK();
}
```

### PayloadFileWriter::Close() (`cpp/src/arrow/ipc/writer.cc:1502`)

This finalizes the IPC file format on the stream but **does not close the
underlying OutputStream**:

```cpp
Status Close() override {
    // Write 0 EOS message for compatibility with sequential readers
    RETURN_NOT_OK(WriteEOS());

    // Write file footer
    RETURN_NOT_OK(UpdatePosition());
    int64_t initial_position = position_;
    RETURN_NOT_OK(
        WriteFileFooter(*schema_, dictionaries_, record_batches_, metadata_, sink_));

    // Write footer length (4 bytes, little-endian)
    RETURN_NOT_OK(UpdatePosition());
    int32_t footer_length = static_cast<int32_t>(position_ - initial_position);
    if (footer_length <= 0) {
        return Status::Invalid("Invalid file footer");
    }
    footer_length = bit_util::ToLittleEndian(footer_length);
    RETURN_NOT_OK(Write(&footer_length, sizeof(int32_t)));

    // Write magic bytes to end file
    return Write(kArrowMagicBytes, strlen(kArrowMagicBytes));
}
```

The `sink_` pointer comes from `StreamBookKeeper` (`writer.cc:1367`), which
stores both a raw pointer (`sink_`) and optionally an owning shared pointer
(`owned_sink_`). Neither `PayloadFileWriter::Close()` nor
`StreamBookKeeper` ever call `sink_->Close()`.

The header `cpp/src/arrow/ipc/writer.h:136` is explicit about this contract:

> "User is responsible for closing the actual OutputStream."

## When Does the File Descriptor Actually Close?

### Case 1: Sink created from a file path string

When a path string is passed to `RecordBatchFileWriter(sink, schema)`, the
Cython `get_writer()` function (`python/pyarrow/io.pxi:2195`) creates a
temporary `OSFile` wrapping a C++ `FileOutputStream`:

```python
cdef get_writer(object source, shared_ptr[COutputStream]* writer):
    # ...
    source = OSFile(source_path, mode='w')
    # ...
    nf = source
    writer[0] = nf.get_output_stream()
```

The `OSFile` Python object is ephemeral -- it goes out of scope immediately.
However, the `shared_ptr<COutputStream>` that was extracted from it is kept alive
inside `StreamBookKeeper::owned_sink_` for the lifetime of the writer.

The fd is closed when:

1. All Python references to the writer are dropped.
2. The `IpcFormatWriter` and its `PayloadFileWriter` are destroyed.
3. The `shared_ptr<COutputStream>` ref count drops to zero.
4. `FileOutputStream::~FileOutputStream()` calls `internal::CloseFromDestructor(this)`
   (`cpp/src/arrow/io/file.cc:357`).
5. That calls `OSFile::Close()` -> `FileDescriptor::Close()`, which closes the fd.

**There can be a window between `writer.close()` and the actual fd close**,
depending on when garbage collection runs.

### Case 2: Sink is a user-owned NativeFile / OSFile

If you pass an already-opened `NativeFile` to the writer, the writer holds a
`shared_ptr` to the same underlying `COutputStream`. The fd remains open as long
as the Python `NativeFile` object is alive. You must close it yourself (or use it
as a context manager).

### Case 3: C++ API with raw pointer overload

`MakeFileWriter(io::OutputStream* sink, ...)` stores only the raw pointer (no
`owned_sink_`). The caller owns the stream entirely and must close it after
calling `RecordBatchWriter::Close()`.

## Summary

| Event                                | What happens                                                    |
|--------------------------------------|-----------------------------------------------------------------|
| `writer.close()`                     | Writes EOS, footer, footer length, magic bytes. **fd stays open.** |
| Writer object is garbage collected   | `shared_ptr<OutputStream>` refcount -> 0, `FileOutputStream` destructor closes fd. |
| User closes the sink explicitly      | fd closed immediately.                                          |

## Recommended Pattern

Use context managers for both the writer and any explicitly opened sink to ensure
deterministic cleanup:

```python
import pyarrow as pa
from pyarrow import ipc

# Option A: pass a path (sink lifecycle is tied to the writer)
with ipc.RecordBatchFileWriter(sink="output.arrow", schema=schema) as writer:
    writer.write_batch(batch)
# At exit: writer.close() is called (footer written).
# fd closes when the writer object is destroyed (usually immediately).

# Option B: explicit sink control
with pa.OSFile("output.arrow", mode="w") as sink:
    with ipc.RecordBatchFileWriter(sink, schema=schema) as writer:
        writer.write_batch(batch)
    # writer.close() writes footer
# sink.close() closes the fd
```
