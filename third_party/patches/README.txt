Local dependency patches

The upstream archive hashes remain pinned in dependencies.lock.json. Each
dependency may also list a repository-contained patch manifest and its SHA256.
tools/bootstrap.py verifies and applies these patches even when extraction is
already stamped complete. Exact before/after source hashes make application
idempotent and reject unexpected local edits. Ordered patches for one file form
a verified hash chain; fresh, partially applied and complete states are supported.
No system patch utility is needed.

llvm-json-capacity.json changes LLVM 20.1.8 JSON Object's initializer-list
constructor to initialize its DenseMap with the known property count. It does
not change public object layout, JSON values, or SDK source. The purpose is to
avoid a 64-bucket allocation for every small generated JSON object. Ordinary
parser construction is unaffected by that first patch. This is a local modification, not an
upstream LLVM release change. LLVM's original license remains in .deps/src.

Validation of compiler memory savings is tracked separately in
artifacts/json-object-capacity; the patch's expected benefit is not itself
proof of whole-model compilation or reduced VNNX allocation.

The llvm-json-compact-* manifests add Object::compact() and invoke it after
each successfully parsed nonempty object. It moves existing keys and values to
a pre-sized DenseMap; it does not reparse or reinterpret values. This also avoids
propagating excessive bucket capacity when parsed tensor metadata is copied.
Duplicate-key handling stays in the original parser before compaction. Object
layout is unchanged. Separate evidence is in artifacts/json-parser-capacity.
