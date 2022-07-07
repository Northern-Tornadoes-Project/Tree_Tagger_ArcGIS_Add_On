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
        private ArcGIS.Desktop.Mapping.Map map;
        private System.Collections.Generic.IEnumerable<RasterLayer> rLayers;


        public TreeTagSelectionWindow()
        {
            InitializeComponent();

            map = MapView.Active.Map;
            rLayers = map.GetLayersAsFlattenedList().OfType<RasterLayer>();

            foreach (var raster in rLayers)
            {
                RasterSelectionBox.Items.Add(raster.Name);

            }
        }

        private async void DoneButtonClicked(object sender, RoutedEventArgs e)
        {
            if (RasterSelectionBox.SelectedItems.Count <= 0) return;

            List<RasterLayer> selectedRasters = new List<RasterLayer>();

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

            string cmdCommand = "";

            await QueuedTask.Run(() =>
            {

                var cellSize = selectedRasters[0].GetRaster().GetMeanCellSize();

                (double, double) imageScale = (cellSize.Item1, cellSize.Item2);
                (double, double) topLeft = (Double.MaxValue, Double.MinValue);
                (double, double) bottomRight = (Double.MinValue, Double.MaxValue);


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

                //System.Diagnostics.Debug.WriteLine(System.IO.Path.GetDirectoryName((new System.Uri(Assembly.GetEntryAssembly().CodeBase)).AbsolutePath));

                var pathExe = "C:/Users/danie/anaconda3/envs/kerasSeg/python.exe";

                var pathPython = "F:/arcGIS/TreeTagger/main.py";

                var pathProject = ArcGIS.Desktop.Core.Project.Current.URI;

                //System.Diagnostics.Debug.WriteLine(ArcGIS.Desktop.Core.Project.Current.URI);


                cmdCommand = string.Format(@"/c """"{0}"" ""{1}"" ""{2}"" ""{3}"" ""{4}"" ""{5}"" ""{6}"" ""{7}"" ""{8}"" ""{9}""", 
                                         pathExe, pathPython, pathProject, imageScale.Item1, imageScale.Item2, topLeft.Item1, topLeft.Item2, bottomRight.Item1, bottomRight.Item2, 0);

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

            var procStartInfo = new System.Diagnostics.ProcessStartInfo("cmd", cmdCommand);

            procStartInfo.RedirectStandardOutput = false;
            procStartInfo.RedirectStandardError = false;
            procStartInfo.UseShellExecute = false;

            //procStartInfo.CreateNoWindow = true;
            System.Diagnostics.Process proc = new System.Diagnostics.Process();
            proc.StartInfo = procStartInfo;
            proc.Start();
            proc.WaitForExit();

            /*string error = proc.StandardError.ReadToEnd();

            if (!string.IsNullOrEmpty(error)) MessageBox.Show(string.Format("Error: {0}", error));*/

            proc.Close();
            proc.Dispose();

            await QueuedTask.Run(() =>
            {
                var pathProject = ArcGIS.Desktop.Core.Project.Current.URI;
                string shapeFilesPath = System.IO.Path.GetDirectoryName(pathProject) + "\\TreeTagger";
                string[] shapeFiles = System.IO.Directory.GetFiles(shapeFilesPath);

                List<string> lineShapeFiles = new List<string>();
                List<string> polygonShapeFiles = new List<string>();

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

                lineShapeFiles.Sort();
                polygonShapeFiles.Sort();

                System.Diagnostics.Debug.WriteLine(lineShapeFiles[lineShapeFiles.Count() - 1]);
                System.Diagnostics.Debug.WriteLine(polygonShapeFiles[polygonShapeFiles.Count() - 1]);

                Uri lineFile = new Uri(lineShapeFiles[lineShapeFiles.Count() - 1]);
                Uri polyFile = new Uri(polygonShapeFiles[polygonShapeFiles.Count() - 1]);

                System.Diagnostics.Debug.WriteLine(lineFile.AbsoluteUri);

                var linesLayer = LayerFactory.Instance.CreateLayer(lineFile, map) as FeatureLayer;
                var polygonLayer = LayerFactory.Instance.CreateLayer(polyFile, map) as FeatureLayer;

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
