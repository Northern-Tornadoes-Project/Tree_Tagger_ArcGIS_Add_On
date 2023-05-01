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
    internal class ShowPolygonReRun : Button
    {

        private PolygonReRun _polygonrerun = null;

        protected override void OnClick()
        {
            //already open?
            if (_polygonrerun != null)
                return;
            _polygonrerun = new PolygonReRun();
            _polygonrerun.Owner = FrameworkApplication.Current.MainWindow;
            _polygonrerun.Closed += (o, e) => { _polygonrerun = null; };
            _polygonrerun.Show();
            //uncomment for modal
            //_polygonrerun.ShowDialog();
        }

    }
}
