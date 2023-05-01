using ArcGIS.Core.CIM;
using ArcGIS.Core.Data;
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
using System.Text;
using System.Threading.Tasks;

namespace TreeTaggerModule
{
    internal class ShowDirectionReRunWindow : Button
    {

        private DirectionReRunWindow _directionrerunwindow = null;

        protected override void OnClick()
        {
            //already open?
            if (_directionrerunwindow != null)
                return;
            _directionrerunwindow = new DirectionReRunWindow();
            _directionrerunwindow.Owner = FrameworkApplication.Current.MainWindow;
            _directionrerunwindow.Closed += (o, e) => { _directionrerunwindow = null; };
            _directionrerunwindow.Show();
            //uncomment for modal
            //_directionrerunwindow.ShowDialog();
        }

    }
}
