# AngelFISH
By Jack Forman from the Munsky Group at Colorado State University

# Overview
Image processing is a fundamental tool of a quantitivative biologist.  
Most image processing code is produced in a one off, for a specific task and 
pipeline are often produced for a single project. The goal of this repository is to
improve this experience by making reusable steps and data structures.

To address this problem we first addressed a few problems:
1) Easily load data
2) Structure for multiple analysis in the same data
3) Record the steps that occur to the data
4) Easily save results and load them later
5) Easy paralized and used on HPC

# Data Structure
The primary structur to facilate the easily handling of data is the receipt. 
The receipt tell you what data you are using, how to load it, and what has/will
be done to it. It is a dictionary with some additional features.
It's basic structure is:
{
    'meta_arguments' - parameters associated with the data (nas_location and analysis_name)
    'step_order' - list of the step names in order of execution
    'dirs' - key directories on where to save results
    'steps' - dictionary with keys refering to the step names and holds the parameter for that step
}

This receipt will be passed from step to step, getting updated, loading the data, and executing the image processing.

the data_loader a parameter in meta_arguments that determine how the data is loaded and saved. It is data type specific. 
To be a valid data type, some sort of tree structure is required. File systems work, as well as, H5 files. The current code 
will enforce a data structure as follows:

DataName
|---- Masks
|---- Analysis 1
    |- Results
    |- Status
    |- Figures
|---- Analysis 2
    |- ...
|---- Analysis 3
    |- ...

load_data(receipt) will return all the data associated with an analysis at any point'

# Basic Steps
Steps can be anything you can dream of.  
The requirements are pretty loose:
1) Must take in a receipt, and a step name  
2) they most have a task name (task are the generic form of steps meaning i might have a segment nuc step and a segment cyto step both of these are from the segment task)
3) The step must update receipt with its task_name and all non-default parameters to receipt['step'][step_name]
4) must add its step_name to receipt['step_order']
5) It must write its results to the results directory
6) It must write a status file saying it is complete to the status dir TODO: remove this in the future

Other then these limitations you can make the steps do whatever you want. I highly encorage the building of guis along side each step
so easy tuning is achievable.

# First Run 
Go to protocols and run documentation notebook. 
Here run the cells of the steps you would like to use in the order that you would like to use them. 
You do not and should not use all of them. 
At the end save the pipeline and then you can explore some of the example to do further downstream analysis.

# Future work.
Makes receipts be steps# AngelFISH  
**By Jack Forman, Munsky Group at Colorado State University**

---

## Overview

Image processing is a fundamental tool in the quantitative biologist's toolkit.  
However, most image processing code is written ad hoc for specific tasks or one-off projects, leading to duplicated effort and limited reusability.

The goal of this repository is to streamline that process by creating **reusable processing steps and standardized data structures**, making it easier to build robust, flexible, and scalable image analysis pipelines.

To accomplish this, the project addresses the following key problems:

1. **Easily load and save image data**
2. **Support multiple analyses on the same dataset**
3. **Track and record all processing steps**
4. **Persist results for easy future access**
5. **Run analyses in parallel or on an HPC cluster**

---

## Data Structure

The central data structure used in this repository is the **receipt**.

A **receipt** is a YAML-based dictionary that captures everything about a data processing pipeline:  
- what data to use  
- how to load it  
- what steps to run  
- where to store results

### Receipt Structure

```yaml
{
    'meta_arguments': {       # Metadata about the dataset
        'nas_location': ...,  # Location on shared storage
        'analysis_name': ...  # Unique name for this analysis
    },
    'step_order': [...],      # List of step names in execution order
    'dirs': {...},            # Key directories (results, status, etc.)
    'steps': {                # Dictionary of step parameters
        'step_name_1': {
            'task_name': ..., # Task type (e.g., "segment", "filter", etc.)
            ...
        },
        ...
    }
}
```

The receipt is passed from step to step, updating as tasks are run and data is processed.

---

### Data Loader

The `data_loader` parameter (defined in `meta_arguments`) determines how data is accessed and saved. It must define a tree-like structure that can map across filesystems (e.g., nested folders) or HDF5 files.

Current enforced structure:

```
DataName/
├── Masks/
├── Analysis_1/
│   ├── Results/
│   ├── Status/
│   └── Figures/
├── Analysis_2/
│   └── ...
├── Analysis_3/
    └── ...
```

Using this structure, the function `load_data(receipt)` will return all relevant data associated with an analysis at any point in the pipeline.

---

## Defining Processing Steps

A **step** represents a unit of work in your image processing pipeline. Steps are flexible and can be custom-designed. They must follow a few conventions:

### Step Requirements

1. Must take in a `receipt` and a `step_name`.
2. Must define a `task_name` that identifies the underlying processing task.
    - For example, both `segment_nuc` and `segment_cyto` might share the task `segment`.
3. Must update `receipt['steps'][step_name]` with the task name and any non-default parameters.
4. Must append its `step_name` to `receipt['step_order']`.
5. Must save results to the appropriate results directory.
6. Must create a status file in the status directory upon completion.  
   *(Note: This may be deprecated in future versions.)*

Beyond these requirements, steps can do anything. You're encouraged to build GUI interfaces alongside steps to make tuning parameters easier and more interactive.

---

## First Run

To get started:

1. Navigate to the `protocols/` directory.
2. Open and run the `documentation.ipynb` notebook.
3. Choose the steps you want to run and execute them in your desired order.
    - You don’t have to use all steps — only what your analysis needs.
4. Save the resulting receipt (i.e., your pipeline).
5. Explore the provided examples for more advanced or downstream analyses.

---

## Future Work

- Allow receipts themselves to be reusable as steps  
- Add support for more data formats and structures  
- Develop hierarchical receipts (e.g., nested datasets within datasets)

Add more data structures
Add more deeper data structures, i.e. multiple datasets nested inside eachother




