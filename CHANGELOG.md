# Changelog


The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and
this project adheres to a 2-part versioning scheme X,Y where a change in X means you are
probably fucked while a change in Y means you may not be fucked.



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
