# Changelog


The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and
this project adheres to a 2-part versioning scheme X,Y where a change in X means you are
probably fucked while a change in Y means you may not be fucked.

## [1.9]
### Added
* Better mac support [#8](https://github.com/tech-ascent/tvm-clj/pull/8)
### Fixed/Changed
* Updated compute layer which is not based on jna.  This gives perf benefits during
host->host copies and allows a simpler cpu layer.  Note that the tvm cpu layer is now
100% compliant with all tensor tests so you can intermix tensor operations and tvm
operations.
