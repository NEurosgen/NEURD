//
//  neuron_cgal_segmentation.cpp
//  cgal_testing
//
//  Created by Brendan Celii on 11/25/18.
//  Copyright © 2018 Brendan Celii. All rights reserved.
//
// Ported to CGAL 6 (2026-06-02): OFF_reader.h -> IO/OFF.h, CGAL::read_OFF ->
// CGAL::IO::read_OFF, built with -std=c++17. Algorithm (CGAL::sdf_values +
// CGAL::segmentation_from_sdf_values) is unchanged from the Docker build, so the
// SDF / segment CSVs are identical to the original. Recovered from git c11f5b9
// (docker/CGAL/cgal_segmentation/). See cgal/README.md.
//
// 2026-07-24: sdf_values' number_of_rays is now overridable via the NEURD_SDF_RAYS
// env var (default 25 == original). ~100% of this call's cost is the per-face ray
// casting inside sdf_values, and that cost is ~linear in the ray count, so lowering
// it (e.g. 12/8) is the main speed lever; unset keeps output byte-identical.

#include "neuron_cgal_segmentation.hpp"

//
//  segmentation_from_sdf_value_example_2.cpp
//  cgal_testing
//
//  Created by Brendan Celii on 11/1/18.
//  Copyright © 2018 Brendan Celii. All rights reserved.
//
#include <Python.h>
//#include "segmentation_from_sdf_value_example_2.hpp"

#if 0
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/mesh_segmentation.h>
#include <CGAL/property_map.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include <CGAL/IO/OFF.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>
#include <vector>
#endif

#if 1
#include <CGAL/IO/OFF.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/mesh_segmentation.h>
#include <CGAL/property_map.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <cstdlib>   // std::getenv / std::atoi (NEURD_SDF_RAYS override)
#endif

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef CGAL::Polyhedron_3<Kernel> Polyhedron;

namespace patch
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef CGAL::Polyhedron_3<Kernel> Polyhedron;


//PyObject *cgal_segmentation(PyObject *self, PyObject *args)
//const char * cgal_segmentation(const char *location,const char * filename,int number_of_clusters,double smoothing_lambda)
int cgal_segmentation(const char* location_with_filename, int number_of_clusters,double smoothing_lambda)
{

    // create and read Polyhedron
    Polyhedron mesh;

    std::string location_with_filename_str(location_with_filename);
    std::string input_file_name = location_with_filename_str + ".off";
    std::ifstream input(input_file_name.c_str());


    /* New way of importing the mesh */
    if (!input)
    {
        std::cerr << "Cannot open file " << std::endl;
        return 2;
    }
    std::vector<K::Point_3> points;
    std::vector< std::vector<std::size_t> > polygons;
    if (!CGAL::IO::read_OFF(input, points, polygons))
    {
        std::cerr << "Error parsing the OFF file " << std::endl;
        return 3;
    }
    CGAL::Polygon_mesh_processing::orient_polygon_soup(points, polygons);
    CGAL::Polygon_mesh_processing::polygon_soup_to_polygon_mesh(points, polygons, mesh);
    if (CGAL::is_closed(mesh) && (!CGAL::Polygon_mesh_processing::is_outward_oriented(mesh)))
        CGAL::Polygon_mesh_processing::reverse_face_orientations(mesh);

    if(mesh.empty()){
        return 4;
    }
    if(( !CGAL::is_triangle_mesh(mesh))){
        return 6;
    }

    // create a property-map for SDF values
    typedef std::map<Polyhedron::Facet_const_handle, double> Facet_double_map;
    Facet_double_map internal_sdf_map;
    boost::associative_property_map<Facet_double_map> sdf_property_map(internal_sdf_map);
    // Compute SDF values. Cone angle + postprocess stay at CGAL's defaults; the
    // number of rays per facet is the dominant cost of this whole call (~100% of it
    // is the per-face ray casting), and it is CGAL's default of 25. Allow it to be
    // lowered via NEURD_SDF_RAYS to trade segmentation detail for speed (sdf cost is
    // ~linear in ray count). Unset (or <=0) => 25 => byte-identical to the default.
    std::size_t number_of_rays = 25;
    if (const char* rays_env = std::getenv("NEURD_SDF_RAYS")) {
        int rays_val = std::atoi(rays_env);
        if (rays_val > 0)
            number_of_rays = static_cast<std::size_t>(rays_val);
    }
    CGAL::sdf_values(mesh, sdf_property_map,
                     2.0 / 3.0 * CGAL_PI, number_of_rays, /*postprocess=*/true);

    char smoothing_lambda_str[5];
    snprintf(smoothing_lambda_str, sizeof(smoothing_lambda_str), "%.2f",smoothing_lambda );


    //print out the sdf values as well


    std::string output_filename_sdf = location_with_filename_str + "-cgal_" + patch::to_string(number_of_clusters) + "_"+smoothing_lambda_str + "_sdf.csv";
    std::ofstream myfile_sdf;
    myfile_sdf.open (output_filename_sdf.c_str());

    for(Polyhedron::Facet_const_iterator facet_it = mesh.facets_begin();
        facet_it != mesh.facets_end(); ++facet_it) {
        myfile_sdf << sdf_property_map[facet_it] << std::endl;
    }

    std::cout << std::endl;
    myfile_sdf.close();


    // create a property-map for segment-ids
    typedef std::map<Polyhedron::Facet_const_handle, std::size_t> Facet_int_map;
    Facet_int_map internal_segment_map;
    boost::associative_property_map<Facet_int_map> segment_property_map(internal_segment_map);
    // segment the mesh using default parameters for number of levels, and smoothing lambda
    // Any other scalar values can be used instead of using SDF values computed using the CGAL function

    std::size_t number_of_segments = CGAL::segmentation_from_sdf_values(mesh, sdf_property_map, segment_property_map, number_of_clusters, smoothing_lambda);
    std::cout << "Number of segments: " << number_of_segments << std::endl;
    std::ofstream myfile;
    //create string for the float


    std::string output_filename = location_with_filename_str + "-cgal_" + patch::to_string(number_of_clusters) + "_"+smoothing_lambda_str;
    std::string output_filename_with_csv = output_filename + ".csv";
    myfile.open (output_filename_with_csv.c_str());
    // print segment-ids
    for(Polyhedron::Facet_const_iterator facet_it = mesh.facets_begin();
        facet_it != mesh.facets_end(); ++facet_it) {
        // ids are between [0, number_of_segments -1]
        //going to write the segments to an output CSV file so that they can be loaded into blender later
        myfile << segment_property_map[facet_it] << std::endl;
    }

    std::cout << std::endl;
    myfile.close();

    // Note that we can use the same SDF values (sdf_property_map) over and over again for segmentation.
    // This feature is relevant for segmenting the mesh several times with different parameters.

    //need to save the segmentation values off here
    return 1;
}

int segmentation_example(){
    std::string location = "./";
    std::string filename = "example_neuron";
    std::string location_with_filename_str = location + filename;

    const double smoothing_array[1] = { 0.04};
    const int clusters_array[1] = {6};

    for( int j = 0; j< 1; j = j + 1){
        for( int i = 0; i < 1; i = i + 1 ){
            cgal_segmentation(location_with_filename_str.c_str(),clusters_array[j],smoothing_array[i]);
        }
    }

    return 100;
};

static PyObject* cgal_segmentation_C(PyObject *self, PyObject *args)
{
    int number_of_clusters;
    double smoothing_lambda;
    const char *location_with_filename;
    int output_int;

    if (!PyArg_ParseTuple(args,"sid",&location_with_filename,&number_of_clusters,&smoothing_lambda))
        return NULL;

    output_int = cgal_segmentation(location_with_filename,number_of_clusters,smoothing_lambda);
    return Py_BuildValue("i",output_int);

}


//has a problem with how this only accepts one aragument
static PyObject* cgal_demo(PyObject *self)
{
    return Py_BuildValue("i",segmentation_example());
}


static PyMethodDef cgal_Segmentation_Methods[] =
{
    { "cgal_segmentation", cgal_segmentation_C, METH_VARARGS, "calculates the mesh segmentation" },
    { "cgal_demo", (PyCFunction)cgal_demo, METH_NOARGS, "runs an example segmentation" },
    { NULL,NULL,0, NULL }

};

static struct PyModuleDef cgal_Segmentation_Module =
{
    PyModuleDef_HEAD_INIT,
    "cgal_Segmentation_Module",
    "CGAL Neuron Segmentation Module",
    -1,
    cgal_Segmentation_Methods
};

PyMODINIT_FUNC PyInit_cgal_Segmentation_Module(void )
{
    return PyModule_Create(&cgal_Segmentation_Module);
}
