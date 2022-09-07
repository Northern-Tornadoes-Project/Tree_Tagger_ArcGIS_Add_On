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
    /// Interaction logic for DirectionReRunWindow.xaml
    /// </summary>
    /// 
    //list of direction grid sizes
    public partial class DirectionReRunWindow : ArcGIS.Desktop.Framework.Controls.ProWindow
    {
        //list of direction grid sizes
        private List<(int, int)> gridSizes = new List<(int, int)>();
        //current arcgis map
        private ArcGIS.Desktop.Mapping.Map map;

        private System.Collections.Generic.IEnumerable<FeatureLayer> fLayers;

        public DirectionReRunWindow()
        {
            InitializeComponent();

            //get current map
            map = MapView.Active.Map;

            //get all raster layers on map
            fLayers = map.GetLayersAsFlattenedList().OfType<FeatureLayer>();

            //add raster names to windows selection box
            foreach (var layer in fLayers)
            {
                vectorSelectBox.Items.Add(layer.Name);
            }
        }

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

        //method for when the done button is pressed
        //sets up the python process to perform tree detection
        private async void DoneButtonClicked(object sender, RoutedEventArgs e)
        {
            //if rasters were selected by name rather than using a polygon
            FeatureLayer selectedVector = null;

            if (vectorSelectBox.SelectedItem == null) return;

            string name = (string)vectorSelectBox.SelectedItem;

            foreach (var layer in fLayers)
            {
                if (layer.Name.Equals(name))
                {
                    selectedVector = layer;
                }
            }

            if (selectedVector == null) return;

            //command to be executed using windows command prompt inorder to call python code, which is required for machine learning
            //ArcGIS does have native support for python, but it doesn't support all the required modules
            string cmdCommand = "";
            string arguments = "";

            //This tell arcgis to run this code on the main application thread
            //it is required whenever you need to access specific data from open arcgis project
            await QueuedTask.Run(() =>
            {

                //cell size is the conversion factor from coords to pixels (usuallly 5cm = 1 pixel)
                (double, double) imageScale = (0.05, 0.05);
                //(double, double) topLeft = (Double.MaxValue, Double.MinValue);
                //(double, double) bottomRight = (Double.MinValue, Double.MaxValue);

                //System.Diagnostics.Debug.WriteLine(System.IO.Path.GetDirectoryName((new System.Uri(Assembly.GetEntryAssembly().CodeBase)).AbsolutePath));
                var pathSource = System.IO.Path.GetDirectoryName((new System.Uri(Assembly.GetEntryAssembly().CodeBase)).AbsolutePath).Replace("%20", " ") + "\\TreeTagger";
                var pathExe = pathSource + "\\TreeTagger_venv\\python.exe";

                var pathPython = pathSource + "\\Python\\direction_rerun.py";

                //gets the path to the currently open arcgis project
                var pathProject = System.IO.Path.GetDirectoryName(ArcGIS.Desktop.Core.Project.Current.URI);

                //System.Diagnostics.Debug.WriteLine(ArcGIS.Desktop.Core.Project.Current.URI);

                //add the image file path and top left corner of each selected raster to cmd command
                string fullSpec = selectedVector.GetPath().AbsolutePath;
                /*CIMDataConnection dataConnection = selectedVector.GetDataConnection();
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
                }*/

                var extend = selectedVector.GetFeatureClass().GetExtent();

                arguments += string.Format(@"{0}|{1}|{2}|{3}|{4}|{5}|{6}", 
                                            fullSpec, extend.XMin, extend.YMax, extend.XMax, extend.YMin, 
                                            imageScale.Item1, imageScale.Item2);

                //formates the start of the cmd command with the required parameters
                foreach (var size in gridSizes)
                {
                    arguments += string.Format(@"|{0}|{1}", size.Item1, size.Item2);
                }

                cmdCommand = string.Format(@"/c """"{0}"" ""{1}"" ""{2}""""", pathExe, pathPython, pathProject);


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
                    if (error.Contains("arcgispro-py3") || error.Contains("arcgis-py3"))
                    {
                        MessageBox.Show("ArcGIS has loaded the wrong python virtual environment, please save your project, restart ArcGIS and try again");
                        return;
                    }
                    MessageBox.Show(string.Format("{0}", error));
                    return;
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
                subDirectories.RemoveAll(x => !x.Contains("Direction_Rerun"));
                subDirectories.Sort();
                string[] shapeFiles = System.IO.Directory.GetFiles(subDirectories[subDirectories.Count() - 1]);

                //list of all direction shape files
                List<string> directionShapeFiles = new List<string>();

                //get all not currently displayed shape files
                try
                {
                    foreach (var file in shapeFiles)
                    {
                        if (file.Contains(".shp") && !file.Contains(".lock") && !file.Contains(".xml"))
                        {
                            if (file.Contains("Directions"))
                            {
                                directionShapeFiles.Add(file);
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
                directionShapeFiles.Sort();

                //System.Diagnostics.Debug.WriteLine(lineShapeFiles[lineShapeFiles.Count() - 1]);
                //System.Diagnostics.Debug.WriteLine(polygonShapeFiles[polygonShapeFiles.Count() - 1]);

                if (directionShapeFiles.Count() == 0)
                {
                    MessageBox.Show("Couldn't find any shape files created by Tree Tagger");
                    return;
                }

                Uri directFile = new Uri(directionShapeFiles[directionShapeFiles.Count() - 1]);

                //add layers to project
                try
                {
                    var directionLayer = LayerFactory.Instance.CreateLayer(directFile, map) as FeatureLayer;
                }
                catch (Exception e)
                {
                    MessageBox.Show(string.Format("Couldn't render shape files: {0}", e.Message));
                    return;
                }

            });


            Close();
        }
    }

    
}
