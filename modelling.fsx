#r "nuget: Microsoft.ML.Vision"
#r "nuget: Microsoft.ML.ImageAnalytics"
#r "nuget: SciSharp.TensorFlow.Redist, 2.3.1"
#r "nuget: SixLabors.ImageSharp"

open System
open System.IO
open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.Vision
open Microsoft.ML.Transforms
open SixLabors.ImageSharp
open SixLabors.ImageSharp.Processing

//Defining the model data types.
[<CLIMutable>]
type ImageData = {
    ImagePath: string
    Label: string
}

[<CLIMutable>]
type processedImageData =
  { ImagePath: string
    Image: byte[]
    Label: string
    LabelAsKey: UInt32 }

//Image loader function.
let loadImagesFromDir path = 

    let files = Directory.GetFiles(path, "*",searchOption=SearchOption.AllDirectories)
    files
    |> Array.filter(fun file -> 
        (Path.GetExtension(file) = ".jpg"))        
    |> Array.map(fun file -> 
        let mutable label = Path.GetFileName(file)
        label <-  Directory.GetParent(file).Name       
        {ImagePath=file; Label=label}
    )

let mlContext = new MLContext()
let rnd = Random(1)
let projectRoot = @"D:\Projects\Machine Learning Exploration\Analytics and MachineLearning\Maize disease Recognition" 
let dataDir = Path.Combine(projectRoot,"imageData")

let imageData = loadImagesFromDir dataDir 

//Transform the images into an Idataview type.
let imageIdv = 
    imageData
    |> mlContext.Data.LoadFromEnumerable 
    |> mlContext.Data.ShuffleRows

//Image Processing pipeline engine.Getting the images and their lables and converting them to bytes.
let imagePrePPipeline =
    EstimatorChain()
        .Append(mlContext.Transforms.Conversion.MapValueToKey(inputColumnName = "Label",
                                                              outputColumnName = "LabelAsKey",
                                                              keyOrdinality = ValueToKeyMappingEstimator.KeyOrdinality.ByValue))
        .Append(mlContext.Transforms.LoadRawImageBytes(outputColumnName = "Image",
                                                       imageFolder = dataDir,
                                                       inputColumnName = "ImagePath"))

let prePetImages =
    imagePrePPipeline.Fit(imageIdv)
        .Transform(imageIdv)

let train,test = 
    prePetImages
    |> (fun data -> 
            let trainValSplit = mlContext.Data.TrainTestSplit(data, testFraction = 0.1)                   
            (trainValSplit.TrainSet,trainValSplit.TestSet))

(*
Data Augmentation 
Defining the various data augmentation functions.
*)

let turnRight (imgBytes: byte[]) =
    use img = Image.Load(imgBytes)
    let newImage = img.Clone(fun img -> img.Rotate(90f) |> ignore)
    use ms = new MemoryStream()
    newImage.SaveAsJpeg(ms)
    ms.ToArray()

let turnLeft (imgBytes: byte[]) =
    use img = Image.Load(imgBytes)
    let newImage = img.Clone(fun img -> img.Rotate(-90f) |> ignore)
    use ms = new MemoryStream()
    newImage.SaveAsJpeg(ms)
    ms.ToArray()

let flipHorizontally (imgBytes: byte[]) =
    use img = Image.Load(imgBytes)
    let newImage = img.Clone(fun img -> img.RotateFlip(RotateMode.None, FlipMode.Horizontal) |> ignore)
    use ms = new MemoryStream()
    newImage.SaveAsJpeg(ms)
    ms.ToArray()

let grayScale (imgBytes: byte[]) =
    use img = Image.Load(imgBytes)
    let newImage = img.Clone(fun img -> img.Grayscale() |> ignore)
    use ms = new MemoryStream()
    newImage.SaveAsJpeg(ms)
    ms.ToArray()

let gaussianSharpen (imgBytes :byte[]) =   
    use img = Image.Load(imgBytes)
    let newImage = img.Clone(fun img -> img.GaussianSharpen() |> ignore)
    use ms = new MemoryStream()
    newImage.SaveAsJpeg(ms)
    ms.ToArray()


//Data Augmentation process helper function for randomly picking images.  
let randomlyAugment (augmentationBase: processedImageData seq) (percentage: float) (augmentation: byte[] -> byte[]) =
    let randomIndexesToAugment =
        let n = Seq.length augmentationBase
        Seq.init (int ((float n) * percentage)) (fun _ -> rnd.Next(0, n - 1))

    let pick (idxs: int seq) (s: seq<'a>) =
        let arr = Array.ofSeq s
        seq { for idx in idxs -> arr[idx] }

    let augmentedImages =
        augmentationBase
        |> pick randomIndexesToAugment
        |> Seq.map (fun img -> { img with Image = augmentation img.Image })            
    augmentedImages

let processedImages = mlContext.Data.CreateEnumerable<processedImageData>(train, reuseRowObject = false)

let augmentedData =
    [ turnRight; turnLeft; flipHorizontally;grayScale;gaussianSharpen] 
    |> Seq.map (randomlyAugment processedImages 0.2)
    |> Seq.concat

//Image Processing pipeline after Augmentaion process.
let postProcessingPipeline =
    mlContext.Transforms.Conversion.MapValueToKey(inputColumnName = "Label",
                                                  outputColumnName = "LabelAsKey",         
                                                  keyOrdinality = ValueToKeyMappingEstimator.KeyOrdinality.ByValue)

let classifierOptions = 
    ImageClassificationTrainer.Options(FeatureColumnName = "Image", 
                                       LabelColumnName = "LabelAsKey", 
                                       TestOnTrainSet = true, 
                                       //ValidationSet = validation,                                  
                                       Arch = ImageClassificationTrainer.Architecture.ResnetV2101,                                                                       
                                       MetricsCallback = Action<ImageClassificationTrainer.ImageClassificationMetrics>(fun x -> printfn "%s" (x.ToString())))
//CNN training pipeline engine.
let trainingPipeline = 
    EstimatorChain()
       //.Append(mlContext.Transforms.Conversion.MapValueToKey("LabelAsKey","Label"))
       .Append(mlContext.MulticlassClassification.Trainers.ImageClassification(classifierOptions))
       .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel","PredictedLabel"))
       
let trainData =
    Seq.concat [ processedImages; augmentedData ] 
    |> fun dataset -> mlContext.Data.LoadFromEnumerable(dataset)
    |> fun idv -> mlContext.Data.ShuffleRows(idv)
    |> fun idv -> postProcessingPipeline.Fit(idv).Transform(idv)

let trainedModel = 
    trainData
    |> trainingPipeline.Fit

let predictions = test |> trainedModel.Transform
let metrics = mlContext.MulticlassClassification.Evaluate(predictions,labelColumnName="LabelAsKey")
printfn "MacroAccurracy: %f | LogLoss: %f" metrics.MacroAccuracy metrics.LogLoss

//Save the trained model
let modelDirectory = Path.Combine(projectRoot,"models")
let modelPath = modelDirectory + "/diseaseClassificationModel.zip"
mlContext.Model.Save(trainedModel, prePetImages.Schema, modelPath)
 


