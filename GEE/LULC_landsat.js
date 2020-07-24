///*****************************************************************************************///
//             SUPERVISED MACHINE LEARNING FOR LAND USE LAND COVER CLASSIFICATION            //
//                              Author: Hiroyuki Yokoi                                       //
//                              Contributors: Alex Chunet                                    //
//                            --         Methodology             --                          //
//                 STEP 1. Load Landsat (for L7, conduct SLC sensor failure rectification)   //
//                 STEP 2. Make training and validation datasets                             //
//                 STEP 3. Apply ML classifiers and models (CART, RF, SVM)                   //
//                 STEP 4. Accuracy Assessment                                               //
//*******************************************************************************************//

//*******************************************************************************************//
//                               1. Perform LULC for 2002                                    //
//*******************************************************************************************//
/////////// 1. Get and create rectified Landsat 7 ///////////////////////////////// 
////// Adjustments of Landsat 7 Scan Line Corrector (SLC) sensor failure ///////////
var bands = ['red', 'green', 'blue', 'nir', 'swir']
var percentile = 30
var imageParams = {min: 0.0, max: 0.3, bands: ['red', 'green', 'blue']}
var landsat7_SurfaceReflectance = ee.ImageCollection("LANDSAT/LE07/C01/T1_SR").select(['B3','B2','B1','B4','B5'], bands);
var landsat5_SurfaceReflectance = ee.ImageCollection("LANDSAT/LT05/C01/T1_SR").select(['B3','B2','B1','B4','B5'], bands);

var image_L7_2002 = landsat7_SurfaceReflectance
.filterBounds(MUTP_road_rail)
.filterDate('2002-01-01', '2002-12-31')
.reduce(ee.Reducer.percentile([percentile]))
.rename(bands)

image_L7_2002 = image_L7_2002.multiply(0.0001)

//.reduce(ee.Reducer.percentile([percentile]))
//.rename(bands)

// remove artifacts at the edges
image_L7_2002 = image_L7_2002.updateMask(image_L7_2002.select(0).mask().focal_min(90, 'circle', 'meters'))

Map.addLayer(image_L7_2002, imageParams, "image_L7_2002_original", false);

var MIN_SCALE = 1;
var MAX_SCALE = 3;
var MIN_NEIGHBORS = 144;

/* Apply the USGS L7 Phase-2 Gap filling protocol, using a single kernel size. */
var GapFill = function(src, fill, kernelSize) {
  var kernel = ee.Kernel.square(kernelSize * 30, "meters", false)
  
  // Find the pixels common to both scenes.
  var common = src.mask().and(fill.mask())
  var fc = fill.updateMask(common)
  var sc = src.updateMask(common)

  // Find the primary scaling factors with a regression.
  // Interleave the bands for the regression.  This assumes the bands have the same names.
  var regress = fc.addBands(sc)
  regress = regress.select(regress.bandNames().sort())
  var fit = regress.reduceNeighborhood(ee.Reducer.linearFit().forEach(src.bandNames()),  kernel, null, false)
  var offset = fit.select(".*_offset")
  var scale = fit.select(".*_scale")

  // Find the secondary scaling factors using just means and stddev
  var reducer = ee.Reducer.mean().combine(ee.Reducer.stdDev(), null, true)
  var src_stats = src.reduceNeighborhood(reducer, kernel, null, false)
  var fill_stats = fill.reduceNeighborhood(reducer, kernel, null, false)
  var scale2 = src_stats.select(".*stdDev").divide(fill_stats.select(".*stdDev"))
  var offset2 = src_stats.select(".*mean").subtract(fill_stats.select(".*mean").multiply(scale2))

  var invalid = scale.lt(MIN_SCALE).or(scale.gt(MAX_SCALE))
  scale = scale.where(invalid, scale2)
  offset = offset.where(invalid, offset2)

  // When all else fails, just use the difference of means as an offset.  
  var invalid2 = scale.lt(MIN_SCALE).or(scale.gt(MAX_SCALE))
  scale = scale.where(invalid2, 1)
  offset = offset.where(invalid2, src_stats.select(".*mean").subtract(fill_stats.select(".*mean")))

  // Apply the scaling and mask off pixels that didn't have enough neighbors.
  var count = common.reduceNeighborhood(ee.Reducer.count(), kernel, null, true, "boxcar")
  var scaled = fill.multiply(scale).add(offset)
      .updateMask(count.gte(MIN_NEIGHBORS))

  return src.unmask(scaled, true)
}

var fill_02 = landsat5_SurfaceReflectance
.filterBounds(MUTP_road_rail)
.filterDate('2001-01-01', '2003-01-01')
//.median()
.reduce(ee.Reducer.percentile([percentile]))
.rename(bands)

fill_02 = fill_02.multiply(0.0001)

Map.addLayer(fill_02, imageParams, "image (fill) Landsat 5 2002", false);

var image_L7_2002 = GapFill(image_L7_2002, fill_02, 10);
print('L7 Rectified Image 2002: ', image_L7_2002)

Map.addLayer(image_L7_2002, imageParams, "L7_2002_filled");
////////////////////////////////////////////////////////////////////////////////////
/////////// 2. Train data ///////////////////////////////// 
// Merge points together
var newfc_2002 = built_up_2002.merge(water_2002).merge(green_2002)//.merge(barren);
print('NewFeatureCollection2002: ', newfc_2002, 'newfc_2002')

// Select the bands for training
var bands = ['red', 'green', 'blue', 'nir',  'swir'];

// This property of the table stores the land cover labels.
var label = 'landcover';

// Assign random numbers in preparation for a test/train split that will maintain class proportions.
var seed = 2002;
var point_2002 = newfc_2002.randomColumn('random', seed);

var regionsOfInterest2002 = image_L7_2002.select(bands).sampleRegions({
	collection: point_2002,
	properties: [label, 'random'],
	scale: 30,
	tileScale: 16
})

var training_2002 = regionsOfInterest2002.filterMetadata('random', 'less_than', 0.7);
var validation_2002 = regionsOfInterest2002.filterMetadata('random', 'not_less_than', 0.7);

/////////// 3. Run Classifier for training and validation (CART, Random Forest, SVM) ///////////////////////////////// 

//////////3-1. CART///////////
// Make a CART classifier.
var classifier_CART_train_2002 = ee.Classifier.cart({randomSeed:2002})

// Train the classifier for training data
var trainingClassifier_CART_2002 = classifier_CART_train_2002.train(training_2002, label, bands)

// Applying the classifier to the validation data
var validated_CART_2002 = validation_2002.classify(trainingClassifier_CART_2002)
var errorMatrix_CART_2002 = validated_CART_2002.errorMatrix(label, 'classification')

// Retrain the classifiers using the full dataset.
var fullClassifier_CART_2002 = classifier_CART_train_2002.train(regionsOfInterest2002, label, bands);

// Classify the image.
var classified_CART_train_2002 = image_L7_2002.select(bands).classify(fullClassifier_CART_2002);

// Print all the relevant matrix
print('L7_2002 CART Training error matrix: ', trainingClassifier_CART_2002.confusionMatrix())
print('L7_2002 CART Training overall accuracy: ', trainingClassifier_CART_2002.confusionMatrix().accuracy())
print('L7_2002 CART Validation error matrix:', errorMatrix_CART_2002)
print('L7_2002 CART Validation overall accuracy:', errorMatrix_CART_2002.accuracy())

//////////3-2. Random Forest///////////
// Make a Random Forest classifier.
var classifier_RF_train_2002 = ee.Classifier.randomForest({
  numberOfTrees:10,
  seed: 1
})

// Train the classifier for training data
var trainingClassifier_RF_2002 = classifier_RF_train_2002.train(training_2002, label, bands)

// Applying the classifier to the validation data
var validated_RF_2002 = validation_2002.classify(trainingClassifier_RF_2002)
var errorMatrix_RF_2002 = validated_RF_2002.errorMatrix(label, 'classification')

// Retrain the classifiers using the full dataset.
var fullClassifier_RF_2002 = classifier_RF_train_2002.train(regionsOfInterest2002, label, bands);

// Classify the image.
var classified_RF_train_2002 = image_L7_2002.select(bands).classify(fullClassifier_RF_2002);

// Print all the relevant matrix
print('L7_2002 RF Training error matrix: ', trainingClassifier_RF_2002.confusionMatrix())
print('L7_2002 RF Training overall accuracy: ', trainingClassifier_RF_2002.confusionMatrix().accuracy())
print('L7_2002 RF Validation error matrix:', errorMatrix_RF_2002)
print('L7_2002 RF Validation overall accuracy:', errorMatrix_RF_2002.accuracy())

// //////////3-3. SVM///////////
// Make an SVM classifier.
var classifier_SVM_train_2002 = ee.Classifier.svm({
  kernelType: 'RBF',
  gamma: 0.5,
  cost: 10
})

// Train the classifier for training data
var trainingClassifier_SVM_2002 = classifier_SVM_train_2002.train(training_2002, label, bands)

// Applying the classifier to the validation data
var validated_SVM_2002 = validation_2002.classify(trainingClassifier_SVM_2002)
var errorMatrix_SVM_2002 = validated_SVM_2002.errorMatrix(label, 'classification')

// Retrain the classifiers using the full dataset.
var fullClassifier_SVM_2002 = classifier_SVM_train_2002.train(regionsOfInterest2002, label, bands);

// Classify the image.
var classified_SVM_train_2002 = image_L7_2002.select(bands).classify(fullClassifier_SVM_2002);

// Print all the relevant matrix
print('L7_2002 SVM Training error matrix: ', trainingClassifier_SVM_2002.confusionMatrix())
print('L7_2002 SVM Training overall accuracy: ', trainingClassifier_SVM_2002.confusionMatrix().accuracy())
print('L7_2002 SVM Validation error matrix:', errorMatrix_SVM_2002)
print('L7_2002 SVM Validation overall accuracy:', errorMatrix_SVM_2002.accuracy())
///////////////////

// // /////////// 4. Display the classification result and the input image ///////////////////////////////// 

// Define a palette for the Land Use classification.
var palette = [
  'D3D3D3', // built_up (0)  // grey
  '0000FF', // water (1)  // blue
  '008000'//, // forest (2) // green
  //'804b00'  // barren (3) // brown
];

// // Display the classification result and the input image.
Map.centerObject(MUTP_road_rail, 11)
// Map.addLayer(classified_CART_train_2002, {min: 0, max: 2, palette: palette}, 'L7_2002 CART LULC');
// Map.addLayer(classified_RF_train_2002, {min: 0, max: 2, palette: palette}, 'L7_2002 RF LULC');
Map.addLayer(classified_SVM_train_2002, {min: 0, max: 2, palette: palette}, 'L7_2002 SVM LULC');

// //*******************************************************************************************//


//*******************************************************************************************//
//                               2. Perform LULC for 2011                                    //
//*******************************************************************************************//
/////////// 1. Get and create rectified Landsat 7 ///////////////////////////////// 
////// Adjustments of Landsat 7 Scan Line Corrector (SLC) sensor failure ///////////
var bands = ['red', 'green', 'blue', 'nir', 'swir']
var percentile = 30
var imageParams = {min: 0.0, max: 0.3, bands: ['red', 'green', 'blue']}
var landsat7_SurfaceReflectance = ee.ImageCollection("LANDSAT/LE07/C01/T1_SR").select(['B3','B2','B1','B4','B5'], bands);
var landsat5_SurfaceReflectance = ee.ImageCollection("LANDSAT/LT05/C01/T1_SR").select(['B3','B2','B1','B4','B5'], bands);

var image_L7 = landsat7_SurfaceReflectance
.filterBounds(MUTP_road_rail)
.filterDate('2011-01-01', '2011-12-31')
.reduce(ee.Reducer.percentile([percentile]))
.rename(bands)

image_L7 = image_L7.multiply(0.0001)

//.reduce(ee.Reducer.percentile([percentile]))
//.rename(bands)

// remove artifacts at the edges
image_L7 = image_L7.updateMask(image_L7.select(0).mask().focal_min(90, 'circle', 'meters'))

Map.addLayer(image_L7, imageParams, "image_L7_original", false);

var MIN_SCALE = 1;
var MAX_SCALE = 3;
var MIN_NEIGHBORS = 144;

/* Apply the USGS L7 Phase-2 Gap filling protocol, using a single kernel size. */
var GapFill = function(src, fill, kernelSize) {
  var kernel = ee.Kernel.square(kernelSize * 30, "meters", false)
  
  // Find the pixels common to both scenes.
  var common = src.mask().and(fill.mask())
  var fc = fill.updateMask(common)
  var sc = src.updateMask(common)

  // Find the primary scaling factors with a regression.
  // Interleave the bands for the regression.  This assumes the bands have the same names.
  var regress = fc.addBands(sc)
  regress = regress.select(regress.bandNames().sort())
  var fit = regress.reduceNeighborhood(ee.Reducer.linearFit().forEach(src.bandNames()),  kernel, null, false)
  var offset = fit.select(".*_offset")
  var scale = fit.select(".*_scale")

  // Find the secondary scaling factors using just means and stddev
  var reducer = ee.Reducer.mean().combine(ee.Reducer.stdDev(), null, true)
  var src_stats = src.reduceNeighborhood(reducer, kernel, null, false)
  var fill_stats = fill.reduceNeighborhood(reducer, kernel, null, false)
  var scale2 = src_stats.select(".*stdDev").divide(fill_stats.select(".*stdDev"))
  var offset2 = src_stats.select(".*mean").subtract(fill_stats.select(".*mean").multiply(scale2))

  var invalid = scale.lt(MIN_SCALE).or(scale.gt(MAX_SCALE))
  scale = scale.where(invalid, scale2)
  offset = offset.where(invalid, offset2)

  // When all else fails, just use the difference of means as an offset.  
  var invalid2 = scale.lt(MIN_SCALE).or(scale.gt(MAX_SCALE))
  scale = scale.where(invalid2, 1)
  offset = offset.where(invalid2, src_stats.select(".*mean").subtract(fill_stats.select(".*mean")))

  // Apply the scaling and mask off pixels that didn't have enough neighbors.
  var count = common.reduceNeighborhood(ee.Reducer.count(), kernel, null, true, "boxcar")
  var scaled = fill.multiply(scale).add(offset)
      .updateMask(count.gte(MIN_NEIGHBORS))

  return src.unmask(scaled, true)
}

var fill = landsat5_SurfaceReflectance
.filterBounds(MUTP_road_rail)
.filterDate('2010-01-01', '2012-01-01')
//.median()
.reduce(ee.Reducer.percentile([percentile]))
.rename(bands)

fill = fill.multiply(0.0001)

Map.addLayer(fill, imageParams, "image (fill) Landsat 5 2011", false);

var image = GapFill(image_L7, fill, 10);
print('L7_2011 Rectified Image: ', image)

Map.addLayer(image, imageParams, "L7_2011_filled");
//////////////////////////////////////////////////////////////////////////////////

/////////// 2. Train data ///////////////////////////////// 
// Merge points together
var newfc_2011 = built_up_2011.merge(water_2011).merge(green_2011)//.merge(barren);
print('NewFeatureCollection2011: ', newfc_2011, 'newfc_2011')

// Select the bands for training
var bands = ['red', 'green', 'blue', 'nir',  'swir'];

// This property of the table stores the land cover labels.
var label = 'landcover';

// Assign random numbers in preparation for a test/train split that will maintain class proportions.
var seed = 2011;
var point_2011 = newfc_2011.randomColumn('random', seed);

var regionsOfInterest2011 = image.select(bands).sampleRegions({
	collection: point_2011,
	properties: [label, 'random'],
	scale: 30,
	tileScale: 16
})

var training_2011 = regionsOfInterest2011.filterMetadata('random', 'less_than', 0.7);
var validation_2011 = regionsOfInterest2011.filterMetadata('random', 'not_less_than', 0.7);

/////////// 3. Run Classifier for training and validation (CART, Random Forest, SVM) ///////////////////////////////// 

// //////////3-1. CART///////////
// Make a CART classifier.
var classifier_CART_train_2011 = ee.Classifier.cart({randomSeed:2011});

// Train the classifier for training data
var trainingClassifier_CART_2011 = classifier_CART_train_2011.train(training_2011, label, bands);

// Applying the classifier to the validation data
var validated_CART_2011 = validation_2011.classify(trainingClassifier_CART_2011)
var errorMatrix_CART_2011 = validated_CART_2011.errorMatrix(label, 'classification')

// Retrain the classifiers using the full dataset.
var fullClassifier_CART_2011 = classifier_CART_train_2011.train(regionsOfInterest2011, label, bands);

// Classify the image.
var classified_CART_train_2011 = image.select(bands).classify(fullClassifier_CART_2011);

// Print all the relevant matrix
print('L7_2011 CART Training error matrix: ', trainingClassifier_CART_2011.confusionMatrix())
print('L7_2011 CART Training overall accuracy: ', trainingClassifier_CART_2011.confusionMatrix().accuracy())
print('L7_2011 CART Validation error matrix:', errorMatrix_CART_2011)
print('L7_2011 CART Validation overall accuracy:', errorMatrix_CART_2011.accuracy())

//////////3-2. Random Forest///////////
// Make a Random Forest classifier.
var classifier_RF_train_2011 = ee.Classifier.randomForest({
  numberOfTrees:10,
  seed: 1
})

// Train the classifier for training data
var trainingClassifier_RF_2011 = classifier_RF_train_2011.train(training_2011, label, bands)

// Applying the classifier to the validation data
var validated_RF_2011 = validation_2011.classify(trainingClassifier_RF_2011)
var errorMatrix_RF_2011 = validated_RF_2011.errorMatrix(label, 'classification')

// Retrain the classifiers using the full dataset.
var fullClassifier_RF_2011 = classifier_RF_train_2011.train(regionsOfInterest2011, label, bands);

// Classify the image.
var classified_RF_train_2011 = image.select(bands).classify(fullClassifier_RF_2011);

// Print all the relevant matrix
print('L7_2011 RF Training error matrix: ', trainingClassifier_RF_2011.confusionMatrix())
print('L7_2011 RF Training overall accuracy: ', trainingClassifier_RF_2011.confusionMatrix().accuracy())
print('L7_2011 RF Validation error matrix:', errorMatrix_RF_2011)
print('L7_2011 RF Validation overall accuracy:', errorMatrix_RF_2011.accuracy())

// //////////3-3. SVM///////////
// Make an SVM classifier.
var classifier_SVM_train_2011 = ee.Classifier.svm({
  kernelType: 'RBF',
  gamma: 0.5,
  cost: 10
})

// Train the classifier for training data
var trainingClassifier_SVM_2011 = classifier_SVM_train_2011.train(training_2011, label, bands)

// Applying the classifier to the validation data
var validated_SVM_2011 = validation_2011.classify(trainingClassifier_SVM_2011)
var errorMatrix_SVM_2011 = validated_SVM_2011.errorMatrix(label, 'classification')

// Retrain the classifiers using the full dataset.
var fullClassifier_SVM_2011 = classifier_SVM_train_2011.train(regionsOfInterest2011, label, bands);

// Classify the image.
var classified_SVM_train_2011 = image.select(bands).classify(fullClassifier_SVM_2011);

// Print all the relevant matrix
print('L7_2011 SVM Training error matrix: ', trainingClassifier_SVM_2011.confusionMatrix())
print('L7_2011 SVM Training overall accuracy: ', trainingClassifier_SVM_2011.confusionMatrix().accuracy())
print('L7_2011 SVM Validation error matrix:', errorMatrix_SVM_2011)
print('L7_2011 SVM Validation overall accuracy:', errorMatrix_SVM_2011.accuracy())

///////////////////
/////////// 4. Display the classification result and the input image ///////////////////////////////// 

// Display the classification result and the input image.
Map.addLayer(classified_CART_train_2011, {min: 0, max: 2, palette: palette}, 'L7_2011 CART LULC');
Map.addLayer(classified_RF_train_2011, {min: 0, max: 2, palette: palette}, 'L7_2011 RF LULC');
Map.addLayer(classified_SVM_train_2011, {min: 0, max: 2, palette: palette}, 'L7_2011 SVM LULC');
//*******************************************************************************************//

//*******************************************************************************************//
//                               3. Perform LULC for 2018                                    //
//*******************************************************************************************//

//*******************************************************************************************//
/////////// 1. Get and create rectified Landsat 7 ///////////////////////////////// 
var L8 = ee.ImageCollection('LANDSAT/LC08/C01/T1_RT')
.filterDate('2018-01-01', '2018-12-31')
.filterBounds(MUTP_road_rail)

var image_L8 = ee.Algorithms.Landsat.simpleComposite({
  collection:L8,
  asFloat:true
})

Map.addLayer(image_L8, {bands:['B4','B3','B2'], min:0, max:0.3}, 'L8_True_Color')

/////////// 2. Train data ///////////////////////////////// 
// Merge points together
var newfc_2011 = built_up_2011.merge(water_2011).merge(green_2011)//.merge(barren);
print('NewFeatureCollection2011: ', newfc_2011, 'newfc_2011')

// Select the bands for training
var bands_L8 = ['B4', 'B3', 'B2', 'B5', 'B6']

// This property of the table stores the land cover labels.
var label = 'landcover';

// Assign random numbers in preparation for a test/train split that will maintain class proportions.
var seed = 2018;
var point_2018 = newfc_2011.randomColumn('random', seed);

var regionsOfInterest2018 = image_L8.select(bands_L8).sampleRegions({
	collection: point_2018,
	properties: [label, 'random'],
	scale: 30,
	tileScale: 16
})

var training_2018 = regionsOfInterest2018.filterMetadata('random', 'less_than', 0.7);
var validation_2018 = regionsOfInterest2018.filterMetadata('random', 'not_less_than', 0.7);

/////////// 3. Run Classifier for training and validation (CART, Random Forest, SVM) ///////////////////////////////// 

// //////////3-1. CART///////////
// Make a CART classifier.
var classifier_CART_train_2018 = ee.Classifier.cart({randomSeed:2018});

// Train the classifier for training data
var trainingClassifier_CART_2018 = classifier_CART_train_2018.train(training_2018, label, bands_L8)

// Applying the classifier to the validation data
var validated_CART_2018 = validation_2018.classify(trainingClassifier_CART_2018)
var errorMatrix_CART_2018 = validated_CART_2018.errorMatrix(label, 'classification')

// Retrain the classifiers using the full dataset.
var fullClassifier_CART_2018 = classifier_CART_train_2018.train(regionsOfInterest2018, label, bands_L8);

// Classify the image.
var classified_CART_train_2018 = image_L8.select(bands_L8).classify(fullClassifier_CART_2018);

// Print all the relevant matrix
print('L7_2018 CART Training error matrix: ', trainingClassifier_CART_2018.confusionMatrix())
print('L7_2018 CART Training overall accuracy: ', trainingClassifier_CART_2018.confusionMatrix().accuracy())
print('L7_2018 CART Validation error matrix:', errorMatrix_CART_2018)
print('L7_2018 CART Validation overall accuracy:', errorMatrix_CART_2018.accuracy())

//////////3-2. Random Forest///////////
// Make a Random Forest classifier.
var classifier_RF_train_2018 = ee.Classifier.randomForest({
  numberOfTrees:10,
  seed: 1
})

// Train the classifier for training data
var trainingClassifier_RF_2018 = classifier_RF_train_2018.train(training_2018, label, bands_L8)

// Applying the classifier to the validation data
var validated_RF_2018 = validation_2018.classify(trainingClassifier_RF_2018)
var errorMatrix_RF_2018 = validated_RF_2018.errorMatrix(label, 'classification')

// Retrain the classifiers using the full dataset.
var fullClassifier_RF_2018 = classifier_RF_train_2018.train(regionsOfInterest2018, label, bands_L8);

// Classify the image.
var classified_RF_train_2018 = image_L8.select(bands_L8).classify(fullClassifier_RF_2018);

// Print all the relevant matrix
print('L8_2018 RF Training error matrix: ', trainingClassifier_RF_2018.confusionMatrix())
print('L8_2018 RF Training overall accuracy: ', trainingClassifier_RF_2018.confusionMatrix().accuracy())
print('L8_2018 RF Validation error matrix:', errorMatrix_RF_2018)
print('L8_2018 RF Validation overall accuracy:', errorMatrix_RF_2018.accuracy())

//////////3-3. SVM///////////
// Make an SVM classifier.
var classifier_SVM_train_2018 = ee.Classifier.svm({
  kernelType: 'RBF',
  gamma: 0.5,
  cost: 10
})

// Train the classifier for training data
var trainingClassifier_SVM_2018 = classifier_SVM_train_2018.train(training_2018, label, bands_L8)

// Applying the classifier to the validation data
var validated_SVM_2018 = validation_2018.classify(trainingClassifier_SVM_2018)
var errorMatrix_SVM_2018 = validated_SVM_2018.errorMatrix(label, 'classification')

// Retrain the classifiers using the full dataset.
var fullClassifier_SVM_2018 = classifier_SVM_train_2018.train(regionsOfInterest2018, label, bands_L8);

// Classify the image.
var classified_SVM_train_2018 = image_L8.select(bands_L8).classify(fullClassifier_SVM_2018);

// Print all the relevant matrix
print('L8_2018 SVM Training error matrix: ', trainingClassifier_SVM_2018.confusionMatrix())
print('L8_2018 SVM Training overall accuracy: ', trainingClassifier_SVM_2018.confusionMatrix().accuracy())
print('L8_2018 SVM Validation error matrix:', errorMatrix_SVM_2018)
print('L8_2018 SVM Validation overall accuracy:', errorMatrix_SVM_2018.accuracy())

// ///////////////////
// /////////// 4. Display the classification result and the input image ///////////////////////////////// 

// // Display the classification result and the input image.
// Map.addLayer(classified_CART_train_2018, {min: 0, max: 2, palette: palette}, 'L8_2018 CART LULC');
// Map.addLayer(classified_RF_train_2018, {min: 0, max: 2, palette: palette}, 'L8_2018 RF LULC');
Map.addLayer(classified_SVM_train_2018, {min: 0, max: 2, palette: palette}, 'L8_2018 SVM LULC');
// //*******************************************************************************************//

////////////////// RESULT /////////////////
// We got the best peformance for:
//      2002 - SVM
//      2011 - Random Forest
//      2018 - SVM
// The following only assesses the above classifier.
////////////////// RESULT END /////////////

/////////// 4. Calculate the areas by land use type per year ///////////////////////////////// 

// Develop a function to calculate the square km of each classification
var area_calculation = function(image02, image11, image18, AOI){
  
  // Clip the image
  var image_clipped02 = image02.clip(AOI)
  var image_clipped11 = image11.clip(AOI)
  var image_clipped18 = image18.clip(AOI)

  //// 2002 image
  // Select builtup (0), water (1), green (2).
  var image_clipped02_builtup = image_clipped02.eq(0);
  var image_clipped02_water = image_clipped02.eq(1);
  var image_clipped02_green = image_clipped02.eq(2);
  
  // Calculate fallowed area by pixel (0 if pixel was not fallowed)
  var areaImageSqM = ee.Image.pixelArea().clip(AOI);
  var areaImageSqKm = areaImageSqM.multiply(0.000001);
  
  // Apply the sqkm to each classification
  var fallowed_area02_builtup = image_clipped02_builtup.multiply(areaImageSqKm);
  var fallowed_area02_water = image_clipped02_water.multiply(areaImageSqKm);
  var fallowed_area02_green = image_clipped02_green.multiply(areaImageSqKm);

  // Calculate total fallowed area in square kilometers by category. 
  // Urban
  var total_area_builtup_02 = fallowed_area02_builtup.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: AOI,
	  scale: 30,
    maxPixels: 1e18
  });
  // Water
  var total_area_water_02 = fallowed_area02_water.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: AOI,
	  scale: 30,
    maxPixels: 1e18
  });
  // Green
  var total_area_green_02 = fallowed_area02_green.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: AOI,
	  scale: 30,
    maxPixels: 1e18
  });
  
  //// 2011 image
  // Select builtup (0), water (1), green (2).
  var image_clipped11_builtup = image_clipped11.eq(0);
  var image_clipped11_water = image_clipped11.eq(1);
  var image_clipped11_green = image_clipped11.eq(2);
  
  // Calculate fallowed area by pixel (0 if pixel was not fallowed)
  var areaImageSqM = ee.Image.pixelArea().clip(AOI);
  var areaImageSqKm = areaImageSqM.multiply(0.000001);
  
  // Apply the sqkm to each classification
  var fallowed_area11_builtup = image_clipped11_builtup.multiply(areaImageSqKm);
  var fallowed_area11_water = image_clipped11_water.multiply(areaImageSqKm);
  var fallowed_area11_green = image_clipped11_green.multiply(areaImageSqKm);

  // Calculate total fallowed area in square kilometers by category. 
  // Urban
  var total_area_builtup_11 = fallowed_area11_builtup.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: AOI,
	  scale: 30,
    maxPixels: 1e18
  });
  // Water
  var total_area_water_11 = fallowed_area11_water.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: AOI,
	  scale: 30,
    maxPixels: 1e18
  });
  // Green
  var total_area_green_11 = fallowed_area11_green.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: AOI,
	  scale: 30,
    maxPixels: 1e18
  });
  
  //// 2018 image
  // Select builtup (0), water (1), green (2).
  var image_clipped18_builtup = image_clipped18.eq(0);
  var image_clipped18_water = image_clipped18.eq(1);
  var image_clipped18_green = image_clipped18.eq(2);
  
  // Calculate fallowed area by pixel (0 if pixel was not fallowed)
  var areaImageSqM = ee.Image.pixelArea().clip(AOI);
  var areaImageSqKm = areaImageSqM.multiply(0.000001);
  
  // Apply the sqkm to each classification
  var fallowed_area18_builtup = image_clipped18_builtup.multiply(areaImageSqKm);
  var fallowed_area18_water = image_clipped18_water.multiply(areaImageSqKm);
  var fallowed_area18_green = image_clipped18_green.multiply(areaImageSqKm);

  // Calculate total fallowed area in square kilometers by category. 
  // Urban
  var total_area_builtup_18 = fallowed_area18_builtup.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: AOI,
	  scale: 30,
    maxPixels: 1e18
  });
  // Water
  var total_area_water_18 = fallowed_area18_water.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: AOI,
	  scale: 30,
    maxPixels: 1e18
  });
  // Green
  var total_area_green_18 = fallowed_area18_green.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: AOI,
	  scale: 30,
    maxPixels: 1e18
  });
  
  // Create a list
  var total_area = ee.List([total_area_builtup_02, total_area_water_02, total_area_green_02, total_area_builtup_11, total_area_water_11, total_area_green_11, total_area_builtup_18, total_area_water_18, total_area_green_18])
  return total_area
}

// 2km buffer zone calculation
var TotalArea_2km = area_calculation(classified_SVM_train_2002,classified_RF_train_2011,classified_SVM_train_2018, MUTP_road_rail)
print('Total_area 2km buffer', TotalArea_2km)

//*******************************************************************************************//

////****************************************************************************//
/////////// 5. Count the number of pixel per year ///////////////////////////////// 
var pixel_count = function(image, AOI1){
  
  // Clip the image
  var image_clipped1 = image.clip(AOI1)

  // Calculate total pixcel observation
  var total_pixel1 = image_clipped1.reduceRegion({
    reducer: ee.Reducer.count(),
    geometry: AOI1,
    scale: 30
  });

  var total_pixel = ee.List([total_pixel1])
  return total_pixel
}

// Apply area calculation function to each buffer zone
var TotalPixel_count = pixel_count(classified_SVM_train_2018, MUTP_road_rail)
print('TotalPixel:', TotalPixel_count)

// Apply area calculation function to each buffer zone

/// 2km buffer zone calculation
// Apply area calculation function to the adopted classifier, i.e. 2011 SVM, 2011 RF, 2018 SVM

////****************************************************************************//

//////// Feature Data Visualization Parameters //
// Create an empty image into which to paint the features, cast to byte.
var empty = ee.Image().byte();

// Paint all the polygon edges with the same number and width, display.
var outline = empty.paint({
  featureCollection: MUTP_road_rail,
  color: 1,
  width: 3
});
Map.addLayer(outline, {palette: 'FF0000'}, 'MUTP_road_rail_2kmBuff');

//////////Export maps ////////////////////////

Export.image.toDrive({
  image: classified_SVM_train_2002.clip(MUTP_road_rail),
  description: 'Mumbai_LULC_SVM_2002',
  region: MUTP_road_rail.geometry().bounds(),
  scale: 30,
  maxPixels: 1e9})

Export.image.toDrive({
  image: classified_SVM_train_2011.clip(MUTP_road_rail),
  description: 'Mumbai_LULC_SVM_2011',
  region: MUTP_road_rail.geometry().bounds(),
  scale: 30,
  maxPixels: 1e9})
  
Export.image.toDrive({
  image: classified_SVM_train_2018.clip(MUTP_road_rail),
  description: 'Mumbai_LULC_SVM_2018',
  region: MUTP_road_rail.geometry().bounds(),
  scale: 30,
  maxPixels: 1e9})

// ////////////////////////////////////////////////////////////////////////
// // Export consufion matrix //
// // var classifier_CART_validation_L8_array = classifier_CART_validation_L8.confusionMatrix();
// // var exportAccuracy = ee.Feature(null, {matrix: classifier_CART_validation_L8_array.array()})

// // // Export the FeatureCollection.
// // Export.table.toDrive({
// //   collection: ee.FeatureCollection(exportAccuracy),
// //   description: 'exportAccuracy',
// //   fileFormat: 'CSV'
// // });


// Export Total Area //
// Create a function to convert a table style
var change_table_format = function(total_area){
  
  var TotalArea_table = ee.FeatureCollection(total_area
                        .map(function(element){
                        return ee.Feature(null,{prop:element})}))
  return TotalArea_table
}

var TotalArea_2km_table = change_table_format(TotalArea_2km)
var TotalPixel_count_table = change_table_format(TotalPixel_count)

// Total land use size
Export.table.toDrive({
  collection: TotalArea_2km_table,
  description:'TotalArea_2km',
  fileFormat: 'CSV'})

// Total Pixel
Export.table.toDrive({
  collection: TotalPixel_count_table,
  description:'TotalPixel_count',
  fileFormat: 'CSV'})