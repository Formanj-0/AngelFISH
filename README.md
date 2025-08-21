# AngelFISH  
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
    'arguments': {       # Metadata about the dataset
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

The `data_loader` parameter (defined in `arguments`) determines how data is accessed and saved. It must define a tree-like structure that can map across filesystems (e.g., nested folders) or HDF5 files.

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




