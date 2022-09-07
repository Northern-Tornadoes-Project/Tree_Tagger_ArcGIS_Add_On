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
using ArcGIS.Core.Internal.CIM;

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
        //list of direction grid sizes
        private List<(int, int)> gridSizes = new List<(int, int)>();



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
            string textBoxText = ((TextBox)sender).Text;

            char newChar = e.Text[0];

            bool foundDot = textBoxText.Contains('.');

            if (!char.IsDigit(newChar))
            {
                if (textBoxText.Length == 0 || newChar != '.' || foundDot)
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
            float lineExtensionSize = 12.8f;
            int equalize = Convert.ToInt32(equalizeCheckBox.IsChecked);

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
            string arguments = "";

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

                if (selectedRasters.Count == 0)
                {
                    MessageBox.Show("invaild raster selection");
                    return;
                }

                //cell size is the conversion factor from coords to pixels (usuallly 5cm = 1 pixel)
                var cellSize = selectedRasters[0].GetRaster().GetMeanCellSize();

                (double, double) imageScale = (cellSize.Item1, cellSize.Item2);
                (double, double) topLeft = (Double.MaxValue, Double.MinValue);
                (double, double) bottomRight = (Double.MinValue, Double.MaxValue);

                //adjust parameters for image scale
                hdDistThreshold /= (float)imageScale.Item1;
                directMergeThreshold /= (float)imageScale.Item1;
                distThreshold /= (float)imageScale.Item1;
                minLengthThreshold /= (float)imageScale.Item1;

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

                ArcGIS.Core.Geometry.Envelope extentedRegion = new ArcGIS.Core.Geometry.EnvelopeBuilderEx(topLeft.Item1 - lineExtensionSize, 
                                                                                                          bottomRight.Item2 - lineExtensionSize, 
                                                                                                          bottomRight.Item1 + lineExtensionSize, 
                                                                                                          topLeft.Item2 + lineExtensionSize, 
                                                                                                          selectedRasters[0].GetSpatialReference()).ToGeometry();

                List<RasterLayer> borderRasters = new List<RasterLayer>();

                foreach(var raster in rLayers)
                {
                    if (!selectedRasters.Contains(raster))
                    {
                        var rExtent = raster.GetRaster().GetExtent();
                        if (rExtent.Intersects(extentedRegion))
                        {
                            borderRasters.Add(raster);
                        }
                    }
                }


                //System.Diagnostics.Debug.WriteLine(System.IO.Path.GetDirectoryName((new System.Uri(Assembly.GetEntryAssembly().CodeBase)).AbsolutePath));
                var pathSource = System.IO.Path.GetDirectoryName((new System.Uri(Assembly.GetEntryAssembly().CodeBase)).AbsolutePath).Replace("%20", " ") + "\\TreeTagger";
                var pathExe = pathSource + "\\TreeTagger_venv\\python.exe";

                var pathPython = pathSource + "\\Python\\main.py";

                //gets the path to the currently open arcgis project
                var pathProject = System.IO.Path.GetDirectoryName(ArcGIS.Desktop.Core.Project.Current.URI);

                //System.Diagnostics.Debug.WriteLine(ArcGIS.Desktop.Core.Project.Current.URI);

                //formates the start of the cmd command with the required parameters
                foreach (var size in gridSizes)
                {
                    arguments += String.Format(@"{0}|{1}|", size.Item1, size.Item2);
                }
                arguments += "?|";

                

                arguments += string.Format(@"{0}|{1}|{2}|{3}|{4}|{5}|{6}|{7}|{8}|{9}|{10}|{11}|{12}|{13}|{14}",
                                         equalize, hdDistThreshold, hdMinSamples, angleThreshold, directAngleThreshold, directMergeThreshold, distThreshold, minLengthThreshold,
                                         imageScale.Item1, imageScale.Item2, topLeft.Item1, topLeft.Item2, bottomRight.Item1, bottomRight.Item2, Convert.ToInt32(isPolygon));

                cmdCommand = string.Format(@"/c """"{0}"" ""{1}"" ""{2}""""", pathExe, pathPython, pathProject);

                //if a polygon was selected, adds all the points of the polygon to the cmd command
                if (isPolygon)
                {
                    var polyPoints = polygon.Points;
                    foreach (var point in polyPoints)
                    {
                        arguments += string.Format(@"|{0}|{1}", point.Coordinate2D.X, point.Coordinate2D.Y);
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

                    arguments += string.Format(@"|{0}|{1}|{2}", fullSpec, extend.XMin, extend.YMax);
                }

                arguments += "|?";

                foreach (var raster in borderRasters)
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

                    arguments += string.Format(@"|{0}|{1}|{2}", fullSpec, extend.XMin, extend.YMax);
                }

                try
                {
                    if (!System.IO.Directory.Exists(pathProject + "\\TreeTagger"))
                    {
                        System.IO.Directory.CreateDirectory(pathProject + "\\TreeTagger");
                    }

                    using (System.IO.StreamWriter writer = new System.IO.StreamWriter(pathProject + "\\TreeTagger\\args.txt", false)) //// true to append data to the file
                    {
                        writer.WriteLine(arguments);
                    }
                }
                catch (System.IO.IOException e)
                {
                    MessageBox.Show("Error writing args file: " + e.Message);
                    return;
                }

            });

            //System.Diagnostics.Debug.WriteLine(cmdCommand);
            try
            {
                //setup windows process to call cmd command
                var procStartInfo = new System.Diagnostics.ProcessStartInfo("cmd", cmdCommand);

                //System.Diagnostics.Debug.WriteLine(procStartInfo.Arguments);

                //ensure output is not redirected
                procStartInfo.RedirectStandardOutput = false;
                procStartInfo.RedirectStandardError = true;
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

                //string output = proc.StandardOutput.ReadToEnd();
                string error = proc.StandardError.ReadToEnd();

                if (!string.IsNullOrEmpty(error))
                {   
                    if(error.Contains("arcgispro-py3") || error.Contains("arcgis-py3"))
                    {
                        MessageBox.Show("ArcGIS has loaded the wrong python virtual environment, please save your project, restart ArcGIS and try again");
                        return;
                    }
                    MessageBox.Show(string.Format("{0}", error));
                }

                //close and dispose of cmd window
                proc.Close();
                proc.Dispose();
            }
            catch (Exception ex)
            {
                MessageBox.Show(String.Format("Error running Tree Tagger python code: cmd command: {0}, arguments: {1}, error {2}", arguments, cmdCommand, ex.Message));
                return;
            }


            //get the newly create shape files containing the lines and polygon regions and add them to the current arcgis project
            await QueuedTask.Run(() =>
            {
                //getting directory paths
                var pathProject = ArcGIS.Desktop.Core.Project.Current.URI;
                string shapeFilesPath = System.IO.Path.GetDirectoryName(pathProject) + "\\TreeTagger";

                List<string> subDirectories = new List<string>(System.IO.Directory.GetDirectories(shapeFilesPath));
                subDirectories.RemoveAll(x => !x.Contains("Results"));
                subDirectories.Sort();
                string[] shapeFiles = System.IO.Directory.GetFiles(subDirectories[subDirectories.Count() - 1]);

                //list of all line shape files
                List<string> lineShapeFiles = new List<string>();
                //list of all refined lines shape files
                List<string> refinedShapeFiles = new List<string>();
                //list of all polygon shape files
                List<string> polygonShapeFiles = new List<string>();
                //list of all direction shape files
                List<string> directionShapeFiles = new List<string>();
                //list of all line vector shape files
                List<string> vectorShapeFiles = new List<string>();

                //get all not currently displayed shape files
                try
                {
                    foreach (var file in shapeFiles)
                    {
                        if (file.Contains(".shp") && !file.Contains(".lock") && !file.Contains(".xml"))
                        {
                            if (file.Contains("Refined"))
                            {
                                refinedShapeFiles.Add(file);
                            }
                            else if (file.Contains("Lines"))
                            {
                                lineShapeFiles.Add(file);
                            }
                            else if (file.Contains("Regions"))
                            {
                                polygonShapeFiles.Add(file);
                            }
                            else if (file.Contains("Directions"))
                            {
                                directionShapeFiles.Add(file);
                            }
                            else if (file.Contains("Vectors"))
                            {
                                vectorShapeFiles.Add(file);
                            }
                        }
                    }
                }
                catch (Exception e)
                {
                    MessageBox.Show(string.Format("Couldn't load Tree Tagger Shape Files from: {0}, {1}", shapeFilesPath, e.Message));
                    return;
                }

                //sort the files and get the latest files (files are time stamped
                lineShapeFiles.Sort();
                polygonShapeFiles.Sort();
                directionShapeFiles.Sort();
                refinedShapeFiles.Sort();
                vectorShapeFiles.Sort();

                //System.Diagnostics.Debug.WriteLine(lineShapeFiles[lineShapeFiles.Count() - 1]);
                //System.Diagnostics.Debug.WriteLine(polygonShapeFiles[polygonShapeFiles.Count() - 1]);

                if (lineShapeFiles.Count() == 0 || polygonShapeFiles.Count() == 0 || directionShapeFiles.Count() == 0 || refinedShapeFiles.Count() == 0 || vectorShapeFiles.Count() == 0)
                {
                    MessageBox.Show("Couldn't find any shape files created by Tree Tagger");
                    return;
                }

                Uri lineFile = new Uri(lineShapeFiles[lineShapeFiles.Count() - 1]);
                Uri polyFile = new Uri(polygonShapeFiles[polygonShapeFiles.Count() - 1]);
                Uri directFile = new Uri(directionShapeFiles[directionShapeFiles.Count() - 1]);
                Uri refinedFile = new Uri(refinedShapeFiles[refinedShapeFiles.Count() - 1]);
                Uri vectorFile = new Uri(vectorShapeFiles[vectorShapeFiles.Count() - 1]);

                //add layers to project
                try
                {
                    GroupLayer groupLayer = LayerFactory.Instance.CreateGroupLayer(map, 0, "TreeTagger");
                    var linesLayer = LayerFactory.Instance.CreateLayer(lineFile, groupLayer) as FeatureLayer;
                    var polygonLayer = LayerFactory.Instance.CreateLayer(polyFile, groupLayer) as FeatureLayer;
                    var directionLayer = LayerFactory.Instance.CreateLayer(directFile, groupLayer) as FeatureLayer;
                    var refinedLayer = LayerFactory.Instance.CreateLayer(refinedFile, groupLayer) as FeatureLayer;
                    var vectorLayer = LayerFactory.Instance.CreateLayer(vectorFile, groupLayer) as FeatureLayer;

                    //format line and polygon shape files to render correctly

                    var lineSymbol = SymbolFactory.Instance.ConstructLineSymbol(ColorFactory.Instance.RedRGB, 2, SimpleLineStyle.Solid);

                    //Get the layer's current renderer
                    CIMSimpleRenderer renderer = linesLayer.GetRenderer() as CIMSimpleRenderer;

                    //Update the symbol of the current simple renderer
                    renderer.Symbol = lineSymbol.MakeSymbolReference();

                    //Update the feature layer renderer
                    linesLayer.SetRenderer(renderer);

                    //Get the layer's current renderer
                    renderer = refinedLayer.GetRenderer() as CIMSimpleRenderer;

                    //Update the symbol of the current simple renderer
                    renderer.Symbol = lineSymbol.MakeSymbolReference();

                    //Update the feature layer renderer
                    refinedLayer.SetRenderer(renderer);

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

                    //CIMGroupLayer cimGroupLayer = new CIMGroupLayer();
                    //cimGroupLayer.Layers = new string[] {lineFile.ToString(), polyFile.ToString(), directFile.ToString(), refinedFile.ToString(), vectorFile.ToString()};
                    

                }
                catch (Exception e)
                {
                    MessageBox.Show(string.Format("Couldn't render shape files: {0}", e.Message));
                    return;
                }

            });

            
            Close();
        }

        private void AddButtonClicked(object sender, RoutedEventArgs e)
        {
            if (!gridSize.Text.Equals("") && !gridMinTrees.Text.Equals(""))
            {
                int gridSizeI = (int)Math.Round(float.Parse(gridSize.Text, System.Globalization.CultureInfo.InvariantCulture));
                int gridMinTreesI = (int)Math.Round(float.Parse(gridMinTrees.Text, System.Globalization.CultureInfo.InvariantCulture));

                gridSizes.Add((gridSizeI, gridMinTreesI));
                gridSizesBox.Items.Add(string.Format("Size: {0}m, Min Trees: {1}", gridSizeI, gridMinTreesI));
            }
        }

        private void RemoveButtonClicked(object sender, RoutedEventArgs e)
        {
            if (gridSizesBox.SelectedIndex != -1)
            {
                gridSizes.RemoveAt(gridSizesBox.SelectedIndex);
                gridSizesBox.Items.RemoveAt(gridSizesBox.SelectedIndex);
            }
        }
    }


}
