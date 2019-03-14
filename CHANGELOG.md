# Changelog



The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and
this project adheres to a 2-part versioning scheme X,Y where a change in X means you are
probably fucked while a change in Y means you may not be fucked.


## [4.6]
We will be staying on major versions now for TVM instead of tracking master.  TVM has
sufficiently matured that tracking master is unnecessary unless we want to make a major
change to the TVM library itself.
### Fixed/Changed
 - tech.compute 3.18
 - tvm release v0.5


## [4.0]
### Fixed/Changed
* Changed to newer resource system.  Mainly naming changes and code reorganization bubbling
up through the stack.

## [3.0]
### Fixed/Changed
 * Moved to newer versions of tech-ascent datatype libraries.
 * TVM things (tensors, AST-nodes, modules, functions, etc) are now both
   scope-based and gc-rooted things.  So the gc can help keep memory and total
   object counts lower especially when you are describing complex systems using
   the api.  Most AST nodes, for instance, when exposed to clojure are not really
   relevant to the larger picture and thus if they happen to be gc'd sooner than
   the resource context winds up then all the better.



## [2.0]
### Added
* example project with dockerfile so anyone can try out the ubuntu version with
only opencv installed.
### Fixed/Changed
#### JNA FTW
Binding layer now dynamically binds to tvm using jna instead of javacpp
1.  Thus we can bind to tvm installed in the system, build by user, or packaged with
jar.  Because tvm has so many system dependencies it makes the most sense for it to be
built specifically for each system (mkl, not mkl, cuda, cudnn, rocm etc) than it does
for us to package the .so file.
2.  Binding layer is split up into many files to make it far easier to understand for
new people to the project.




## [1.4]
### Added
* Better mac support [#8](https://github.com/tech-ascent/tvm-clj/pull/8)
### Fixed/Changed
* Updated compute layer which is not based on jna.  This gives perf benefits during
host->host copies and allows a simpler cpu layer.  Note that the tvm cpu layer is now
100% compliant with all tensor tests so you can intermix tensor operations and tvm
operations.
