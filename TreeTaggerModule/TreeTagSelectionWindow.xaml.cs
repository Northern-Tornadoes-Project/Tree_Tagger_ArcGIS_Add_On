/* Main code for tree tagger add-on
 * 
 * Author: Daniel Butt NTP 2022
 * 
 * 
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Text.RegularExpressions;
using ArcGIS.Core.CIM;
using ArcGIS.Desktop.Mapping;
using ArcGIS.Desktop.Framework.Threading.Tasks;

namespace TreeTaggerModule
{
    /// <summary>
    /// Interaction logic for TreeTagSelectionWindow.xaml
    /// </summary>
    public partial class TreeTagSelectionWindow : ArcGIS.Desktop.Framework.Controls.ProWindow
    {
        //current arcgis map
        private ArcGIS.Desktop.Mapping.Map map;
        //all raster layers on map
        private System.Collections.Generic.IEnumerable<RasterLayer> rLayers;
        //selected ploygon region
        private ArcGIS.Core.Geometry.Polygon polygon;


        public TreeTagSelectionWindow(ArcGIS.Core.Geometry.Polygon poly)
        {
            polygon = poly;

            InitializeComponent();

            //get current map
            map = MapView.Active.Map;

            //get all raster layers on map
            rLayers = map.GetLayersAsFlattenedList().OfType<RasterLayer>();

            //add raster names to windows selection box
            foreach (var raster in rLayers)
            {
                RasterSelectionBox.Items.Add(raster.Name);

            }
            
            //disable selection box if a polygon region was already selected
            if(polygon != null)
            {
                RasterSelectionBox.IsEnabled = false;
            }
        }

        //used to ensure you can only enter a positive floating point number into the textboxes
        private void NumberValidationTextBox(object sender, TextCompositionEventArgs e)
        {
            String text = e.Text;

            bool foundDot = false;

            foreach(char c in text)
            {
                if (c == '.') { 
                    if(foundDot == true)
                    {
                        e.Handled = true;
                        return;
                    }
                    foundDot = true;
                }
                else if (!Char.IsDigit(c))
                {
                    e.Handled = true;
                    return;
                }
                
            }

            e.Handled = false;

        }

        //method for when the done button is pressed
        //sets up the python process to perform tree detection
        private async void DoneButtonClicked(object sender, RoutedEventArgs e)
        {
            //getting paramters from textboxes
            float hdDistThreshold = float.Parse(hdDist.Text, System.Globalization.CultureInfo.InvariantCulture);
            float hdMinSamples = float.Parse(hdMin.Text, System.Globalization.CultureInfo.InvariantCulture);
            float angleThreshold = float.Parse(angleThres.Text, System.Globalization.CultureInfo.InvariantCulture);
            float directAngleThreshold = float.Parse(directAngleThres.Text, System.Globalization.CultureInfo.InvariantCulture);
            float directMergeThreshold = float.Parse(directMergeThres.Text, System.Globalization.CultureInfo.InvariantCulture);
            float distThreshold = float.Parse(distThres.Text, System.Globalization.CultureInfo.InvariantCulture);
            float minLengthThreshold = float.Parse(minLength.Text, System.Globalization.CultureInfo.InvariantCulture);

            //whether a polygon region was selected
            bool isPolygon = (polygon != null);

            //if rasters were selected by name rather than using a polygon
            List<RasterLayer> selectedRasters = new List<RasterLayer>();
            if (!isPolygon)
            {
                if (RasterSelectionBox.SelectedItems.Count <= 0) return;

                foreach (string name in RasterSelectionBox.SelectedItems)
                {
                    foreach (var raster in rLayers)
                    {
                        if (raster.Name.Equals(name))
                        {
                            selectedRasters.Add(raster);
                        }
                    }
                }
            }

            //command to be executed using windows command prompt inorder to call python code, which is required for machine learning
            //ArcGIS does have native support for python, but it doesn't support all the required modules
            string cmdCommand = "";

            //This tell arcgis to run this code on the main application thread
            //it is required whenever you need to access specific data from open arcgis project
            await QueuedTask.Run(() =>
            {
                //if a polygon was selected, get all rasters which intersect the polygon
                if (isPolygon)
                {
                    var geoExtent = polygon.Extent;

                    foreach (var raster in rLayers)
                    {
                        var rExtent = raster.GetRaster().GetExtent();

                        if (rExtent.Intersects(geoExtent))
                        {
                            selectedRasters.Add(raster);
                        }
                    }
                }

                //cell size is the conversion factor from coords to pixels (usuallly 5cm = 1 pixel)
                var cellSize = selectedRasters[0].GetRaster().GetMeanCellSize();

                (double, double) imageScale = (cellSize.Item1, cellSize.Item2);
                (double, double) topLeft = (Double.MaxValue, Double.MinValue);
                (double, double) bottomRight = (Double.MinValue, Double.MaxValue);

                //finds the top left and bottom right corner
                foreach (RasterLayer layer in selectedRasters)
                {
                    var extend = layer.GetRaster().GetExtent();
                    (double, double) layerCoordsTopLeft = (extend.XMin, extend.YMax);
                    (double, double) layerCoordsBottomRight = (extend.XMax, extend.YMin);

                    if (layerCoordsTopLeft.Item1 < topLeft.Item1) topLeft.Item1 = layerCoordsTopLeft.Item1;

                    if (layerCoordsTopLeft.Item2 > topLeft.Item2) topLeft.Item2 = layerCoordsTopLeft.Item2;

                    if (layerCoordsBottomRight.Item1 > bottomRight.Item1) bottomRight.Item1 = layerCoordsBottomRight.Item1;

                    if (layerCoordsBottomRight.Item2 < bottomRight.Item2) bottomRight.Item2 = layerCoordsBottomRight.Item2;

                }

                //if a polygon region was selected, this makes sure the top left/bottom right includes the polygon
                if (isPolygon)
                {
                    var polyPoints = polygon.Points;
                    foreach (var point in polyPoints)
                    {
                        if (point.Coordinate2D.X < topLeft.Item1) topLeft.Item1 = point.Coordinate2D.X;
                        else if (point.Coordinate2D.X > bottomRight.Item1) bottomRight.Item1 = point.Coordinate2D.X;

                        if (point.Coordinate2D.Y > topLeft.Item2) topLeft.Item2 = point.Coordinate2D.Y;
                        else if (point.Coordinate2D.Y < bottomRight.Item2) bottomRight.Item2 = point.Coordinate2D.Y;
                    }
                }


                //System.Diagnostics.Debug.WriteLine(System.IO.Path.GetDirectoryName((new System.Uri(Assembly.GetEntryAssembly().CodeBase)).AbsolutePath));
                var pathExe = "C:/Users/dbutt7/Anaconda3/envs/TreeTaggerSegmentation/python.exe";

                var pathPython = "F:/arcGIS/TreeTagger/main.py";

                //gets the path to the currently open arcgis project
                var pathProject = ArcGIS.Desktop.Core.Project.Current.URI;

                //System.Diagnostics.Debug.WriteLine(ArcGIS.Desktop.Core.Project.Current.URI);

                //formates the start of the cmd command with the required parameters
                cmdCommand = string.Format(@"/c """"{0}"" ""{1}"" ""{2}"" ""{3}"" ""{4}"" ""{5}"" ""{6}"" ""{7}"" ""{8}"" ""{9}"" ""{10}"" ""{11}"" ""{12}"" ""{13}"" ""{14}"" ""{15}"" ""{16}""",
                                         pathExe, pathPython, hdDistThreshold, hdMinSamples, angleThreshold, directAngleThreshold, directMergeThreshold, distThreshold, minLengthThreshold,
                                         pathProject, imageScale.Item1, imageScale.Item2, topLeft.Item1, topLeft.Item2, bottomRight.Item1, bottomRight.Item2, Convert.ToInt32(isPolygon));

                //if a polygon was selected, adds all the points of the polygon to the cmd command
                if (isPolygon)
                {
                    var polyPoints = polygon.Points;
                    foreach (var point in polyPoints)
                    {
                        cmdCommand += string.Format(" \"{0}\" \"{1}\"", point.Coordinate2D.X, point.Coordinate2D.Y);
                    }
                }

                //add the image file path and top left corner of each selected raster to cmd command
                foreach (var raster in selectedRasters)
                {
                    string fullSpec = string.Empty;
                    CIMDataConnection dataConnection = raster.GetDataConnection();
                    if (dataConnection is CIMStandardDataConnection)
                    {
                        CIMStandardDataConnection dataSConnection = dataConnection as CIMStandardDataConnection;

                        string sConnection = dataSConnection.WorkspaceConnectionString;

                        var wFactory = dataSConnection.WorkspaceFactory;
                        if (wFactory == WorkspaceFactory.Raster)
                        {
                            string sWorkspaceName = sConnection.Split('=')[1];

                            string sTable = dataSConnection.Dataset;

                            fullSpec = System.IO.Path.Combine(sWorkspaceName, sTable);
                        }
                    }

                    var extend = raster.GetRaster().GetExtent();

                    cmdCommand += string.Format(@" ""{0}"" ""{1}"" ""{2}""", fullSpec, extend.XMin, extend.YMax);
                }

                cmdCommand += @"""";
            });

            System.Diagnostics.Debug.WriteLine(cmdCommand);

            //setup windows process to call cmd command
            var procStartInfo = new System.Diagnostics.ProcessStartInfo("cmd", cmdCommand);

            //ensure output is not redirected
            procStartInfo.RedirectStandardOutput = false;
            procStartInfo.RedirectStandardError = false;
            procStartInfo.UseShellExecute = false;

            //ensure a window is shown
            procStartInfo.CreateNoWindow = false;

            //create process
            System.Diagnostics.Process proc = new System.Diagnostics.Process();
            proc.StartInfo = procStartInfo;

            //start cmd process and execute command
            proc.Start();

            //wait untill python code has finished
            proc.WaitForExit();

            //string error = proc.StandardError.ReadToEnd();

            //if (!string.IsNullOrEmpty(error)) MessageBox.Show(string.Format("Error: {0}", error));

            //close and dispose of cmd window
            proc.Close();
            proc.Dispose();


            //get the newly create shape files containing the lines and polygon regions and add them to the current arcgis project
            await QueuedTask.Run(() =>
            {
                //getting directory paths
                var pathProject = ArcGIS.Desktop.Core.Project.Current.URI;
                string shapeFilesPath = System.IO.Path.GetDirectoryName(pathProject) + "\\TreeTagger";
                string[] shapeFiles = System.IO.Directory.GetFiles(shapeFilesPath);

                //list of all line shape files
                List<string> lineShapeFiles = new List<string>();
                //list of all polygon shape files
                List<string> polygonShapeFiles = new List<string>();

                //get all not currently displayed shape files
                foreach (var file in shapeFiles)
                {
                    if (file.Contains(".shp") && !file.Contains(".lock"))
                    {
                        if (file.Contains("Lines"))
                        {
                            lineShapeFiles.Add(file);
                        }
                        else if (file.Contains("Regions"))
                        {
                            polygonShapeFiles.Add(file);
                        }
                    }
                }

                //sort the files and get the latest files (files are time stamped
                lineShapeFiles.Sort();
                polygonShapeFiles.Sort();

                System.Diagnostics.Debug.WriteLine(lineShapeFiles[lineShapeFiles.Count() - 1]);
                System.Diagnostics.Debug.WriteLine(polygonShapeFiles[polygonShapeFiles.Count() - 1]);

                Uri lineFile = new Uri(lineShapeFiles[lineShapeFiles.Count() - 1]);
                Uri polyFile = new Uri(polygonShapeFiles[polygonShapeFiles.Count() - 1]);

                System.Diagnostics.Debug.WriteLine(lineFile.AbsoluteUri);

                //add layers to project
                var linesLayer = LayerFactory.Instance.CreateLayer(lineFile, map) as FeatureLayer;
                var polygonLayer = LayerFactory.Instance.CreateLayer(polyFile, map) as FeatureLayer;

                //format line and polygon shape files to render correctly

                var lineSymbol = SymbolFactory.Instance.ConstructLineSymbol(ColorFactory.Instance.RedRGB, 2, SimpleLineStyle.Solid);

                //Get the layer's current renderer
                CIMSimpleRenderer renderer = linesLayer.GetRenderer() as CIMSimpleRenderer;

                //Update the symbol of the current simple renderer
                renderer.Symbol = lineSymbol.MakeSymbolReference();

                //Update the feature layer renderer
                linesLayer.SetRenderer(renderer);


                CIMStroke outline = SymbolFactory.Instance.ConstructStroke(
                     ColorFactory.Instance.CreateRGBColor(255, 0, 255, 30), 2.0, SimpleLineStyle.Solid);
                CIMPolygonSymbol fillWithOutline = SymbolFactory.Instance.ConstructPolygonSymbol(
                     ColorFactory.Instance.CreateRGBColor(255, 0, 255, 30), SimpleFillStyle.Solid, outline);
                //Get the layer's current renderer
                CIMSimpleRenderer rendererp = polygonLayer.GetRenderer() as CIMSimpleRenderer;

                //Update the symbol of the current simple renderer
                rendererp.Symbol = fillWithOutline.MakeSymbolReference();

                //Update the feature layer renderer
                polygonLayer.SetRenderer(rendererp);

            });


            Close();
        }
    }


}
