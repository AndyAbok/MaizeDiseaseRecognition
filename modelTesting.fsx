#r "nuget:Microsoft.ML.Vision"
#r "nuget:Microsoft.ML.ImageAnalytics"
#r "nuget: SciSharp.TensorFlow.Redist, 2.3.1"
#r "nuget: SixLabors.ImageSharp"

open System
open System.IO
open Microsoft.ML
open Microsoft.ML.Data

type ModelInputImage () =
    [<DefaultValue>]
    [<LoadColumn(0)>]
    val mutable public Image :byte[]

    [<DefaultValue>]
    [<LoadColumn(1)>]
    val mutable public Label :string

type ModelOutPutImage () =
    [<DefaultValue>]
    val mutable public PredictedLabel :string

let mlContext = new MLContext()

let predictionEngine = 
    let savedmodelpath = "./models/diseaseClassificationModel.zip" 
    use fsRead = new FileStream(savedmodelpath,FileMode.Open, FileAccess.Read, FileShare.Read)
    let trainedModel,inputSchema = mlContext.Model.Load(fsRead)
    let predictorReloaded = mlContext.Model.CreatePredictionEngine<ModelInputImage, ModelOutPutImage>(trainedModel)
    predictorReloaded

let predictionFunction testdir = 
    let getImage =
        testdir
        |> File.ReadAllBytes

    let toBase64 = Convert.ToBase64String getImage
    let predInput = ModelInputImage()
    predInput.Image <- getImage
    predInput.Label <- null 
    
    let output = predictionEngine.Predict predInput
    output

let testDataDir = Directory.GetFiles("D:/Projects/Machine Learning Exploration/Analytics and MachineLearning/Maize disease Recognition/Test data")

let predictions = 
    testDataDir
    |> Array.map predictionFunction
    |> Seq.iter(fun pred -> 
        if pred.PredictedLabel = "Healthy" 
            then printfn "This Plant is %s" pred.PredictedLabel 
        else
                 printfn "This Plant has the %s disease" pred.PredictedLabel)

predictions


