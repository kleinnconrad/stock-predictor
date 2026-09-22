# Changelog

## [1.5.0](https://github.com/kleinnconrad/stock-predictor/compare/v1.4.0...v1.5.0) (2026-09-22)


### Features

* added rolling (1m, 3m, 6m) uplift diagnostic ([81f7cf3](https://github.com/kleinnconrad/stock-predictor/commit/81f7cf35ce7fb978d7631ba9f861d15b7675c0ec))


### Bug Fixes

* add historic data ([e3c8dc9](https://github.com/kleinnconrad/stock-predictor/commit/e3c8dc9e47c3e3b8f6c394f3c66d908a3304c061))
* artefact constraint for pages deployment ([692a7ef](https://github.com/kleinnconrad/stock-predictor/commit/692a7ef414333baca46b9a78944641ca5488585d))
* include actual legacy results for the uplift chart ([6e365a5](https://github.com/kleinnconrad/stock-predictor/commit/6e365a53c52624dca452f6e18df2915064579f40))
* uplift visual in dashboard ([c247cfb](https://github.com/kleinnconrad/stock-predictor/commit/c247cfb85370f26b8925c2f53892d83af7d70279))

## [1.4.0](https://github.com/kleinnconrad/stock-predictor/compare/v1.3.1...v1.4.0) (2026-09-16)


### Features

* switching to pyproject.toml ([1118e23](https://github.com/kleinnconrad/stock-predictor/commit/1118e23663dbe58fcb96b8a2cd9f398e989761a1))

## [1.3.1](https://github.com/kleinnconrad/stock-predictor/compare/v1.3.0...v1.3.1) (2026-07-23)


### Bug Fixes

* leakage of absolute target ticker values (Open, High, Low, Close, Volume) into the predictor set. Included those vars into the feature engineering turning them stationary. ([9ce4fbd](https://github.com/kleinnconrad/stock-predictor/commit/9ce4fbd2a007459f7fe1446018be66f00b2dcf40))

## [1.3.0](https://github.com/kleinnconrad/stock-predictor/compare/v1.2.1...v1.3.0) (2026-07-16)


### Features

* save 20260716_final_batch_results.json for model validation ([291d069](https://github.com/kleinnconrad/stock-predictor/commit/291d069e9499e3b3771dba5133954106b64d6a79))
