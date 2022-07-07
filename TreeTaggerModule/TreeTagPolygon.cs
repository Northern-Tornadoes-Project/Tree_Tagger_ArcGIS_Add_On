using ArcGIS.Core.CIM;
using ArcGIS.Core.Data;
using ArcGIS.Core.Data.Raster;
using ArcGIS.Core.Geometry;
using ArcGIS.Desktop.Catalog;
using ArcGIS.Desktop.Core;
using ArcGIS.Desktop.Editing;
using ArcGIS.Desktop.Extensions;
using ArcGIS.Desktop.Framework;
using ArcGIS.Desktop.Framework.Contracts;
using ArcGIS.Desktop.Framework.Dialogs;
using ArcGIS.Desktop.Framework.Threading.Tasks;
using ArcGIS.Desktop.Layouts;
using ArcGIS.Desktop.Mapping;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace TreeTaggerModule{
    internal class TreeTagPolygon : MapTool{
        public TreeTagPolygon(){
            IsSketchTool = true;
            SketchType = SketchGeometryType.Polygon;
            SketchOutputMode = SketchOutputMode.Map;
        }

        protected override Task OnToolActivateAsync(bool active){
            return base.OnToolActivateAsync(active);
        }

        protected override async Task<bool> OnSketchCompleteAsync(Geometry geometry){
            Polygon selectedPolygon = (Polygon)geometry;

            var map = MapView.Active.Map;

            string output = await QueuedTask.Run(() =>{
                var geoExtent = selectedPolygon.Extent;
                var rLayers = map.GetLayersAsFlattenedList().OfType<RasterLayer>();

                string str = "null";

                RasterLayer selectedRaster = null;

                foreach (var raster in rLayers){
                    if (!raster.IsVisible) continue;

                    var rExtent = raster.GetRaster().GetExtent();

                    if (rExtent.YMax >= geoExtent.YMax && rExtent.YMin <= geoExtent.YMin && rExtent.XMax >= geoExtent.XMax && rExtent.XMin <= geoExtent.XMin){
                        selectedRaster = raster;
                        break;
                    }
                }

                if (selectedRaster != null){
                    var top_left = selectedRaster.GetRaster().MapToPixel(geoExtent.XMin, geoExtent.YMax);
                    var bottom_right = selectedRaster.GetRaster().MapToPixel(geoExtent.XMax, geoExtent.YMin);
                    var width = bottom_right.Item1 - top_left.Item1;
                    var height = bottom_right.Item2 - top_left.Item2;

                    //call python code
                    /*var pathProExe = System.IO.Path.GetDirectoryName((new System.Uri(Assembly.GetEntryAssembly().CodeBase)).AbsolutePath);

                    if (pathProExe == null) return "null";

                    pathProExe = Uri.UnescapeDataString(pathProExe);*/

                    var pathProExe = "C:/Users/danie/anaconda3/envs/kerasSeg";

                    System.Diagnostics.Debug.WriteLine(pathProExe);

                    /*var pathPython = System.IO.Path.GetDirectoryName((new System.Uri(Assembly.GetEntryAssembly().CodeBase)).AbsolutePath);

                    if (pathPython == null) return "null";

                    pathPython = Uri.UnescapeDataString(pathPython);

                    pathPython = System.IO.Path.Combine(pathPython, @"Python\TreeTagger");*/

                    var pathPython = "D:/arcGIS/TreeTagger";

                    System.Diagnostics.Debug.WriteLine(pathPython);

                    string fullSpec = string.Empty;
                    CIMDataConnection dataConnection = selectedRaster.GetDataConnection();
                    if (dataConnection is CIMStandardDataConnection){
                        CIMStandardDataConnection dataSConnection = dataConnection as CIMStandardDataConnection;

                        string sConnection = dataSConnection.WorkspaceConnectionString;

                        var wFactory = dataSConnection.WorkspaceFactory;
                        if (wFactory == WorkspaceFactory.Raster){
                            string sWorkspaceName = sConnection.Split('=')[1];

                            string sTable = dataSConnection.Dataset;

                            fullSpec = System.IO.Path.Combine(sWorkspaceName, sTable);
                        }
                    }

                    var points = selectedPolygon.Points;
                    String pointArgs = "";

                    foreach (var point in points) {
                        var pointcoord = selectedRaster.GetRaster().MapToPixel(point.Coordinate2D.X, point.Coordinate2D.Y);

                        pointArgs += string.Format(" \"{0}\" \"{1}\"", pointcoord.Item1, pointcoord.Item2);
                    }

                    pointArgs += @"""";

                    var myCommand = string.Format(@"/c """"{0}"" ""{1}"" ""{2}"" ""{3}"" ""{4}"" ""{5}"" ""{6}"" ""{7}"" ""{8}"" ""{9}""",
                        System.IO.Path.Combine(pathProExe, "python.exe"),
                        System.IO.Path.Combine(pathPython, "main.py"),
                        fullSpec,
                        top_left.Item1,
                        top_left.Item2,
                        width,
                        height,
                        geoExtent.XMin,
                        geoExtent.YMax,
                        selectedRaster.GetRaster().GetMeanCellSize().Item1); 

                    myCommand += pointArgs;

                    System.Diagnostics.Debug.WriteLine(myCommand);

                    var procStartInfo = new System.Diagnostics.ProcessStartInfo("cmd", myCommand);

                    //procStartInfo.RedirectStandardOutput = true;
                    //procStartInfo.RedirectStandardError = true;
                    procStartInfo.UseShellExecute = false;

                    //procStartInfo.CreateNoWindow = true;

                    System.Diagnostics.Process proc = new System.Diagnostics.Process();
                    proc.StartInfo = procStartInfo;
                    proc.Start();
                    proc.WaitForExit();
                    proc.Close();

                    //string result = proc.StandardOutput.ReadToEnd();
                    //string error = proc.StandardError.ReadToEnd();

                    //if (!string.IsNullOrEmpty(error)) return string.Format("Error: {0}", error);

                    var filteredRasterLayer = (RasterLayer)LayerFactory.Instance.CreateLayer(new Uri(@"F:\arcGIS\TreeTagger\TreeMask.tif"), map, layerName:"Tree Mask");
                    /*filteredRasterLayer.GetRaster().SetSpatialReference(map.SpatialReference);
                    filteredRasterLayer.GetRaster().SetExtent(EnvelopeBuilder.CreateEnvelope(geoExtent.XMin, geoExtent.YMin, geoExtent.XMax, geoExtent.YMax, map.SpatialReference));

                    FileSystemConnectionPath connectionPath = new FileSystemConnectionPath(new System.Uri(@"D:\arcGIS\TreeTagger"), FileSystemDatastoreType.Raster);
                    FileSystemDatastore dataStore = new FileSystemDatastore(connectionPath);

                    RasterStorageDef rasterStorageDef = new RasterStorageDef();
                    rasterStorageDef.SetPyramidLevel(-1);

                    RasterDataset FinalRasterDataset = filteredRasterLayer.GetRaster().SaveAs("TreeMask.tif", dataStore, "TIFF", rasterStorageDef);

                    connectionPath = null;

                    Raster finalRaster = FinalRasterDataset.CreateFullRaster();*/

                }

                return str;

            });

            //MessageBox.Show(output);

            return true;
        }
    }
}
