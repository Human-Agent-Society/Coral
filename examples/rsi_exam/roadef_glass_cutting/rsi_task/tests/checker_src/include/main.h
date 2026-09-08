#ifndef MAIN_H_INCLUDED
#define MAIN_H_INCLUDED

// Static Path to global_parameters, X_batch, X_defects and X_solution files
#define PATH_TO_INSTANCES           "instances_checker/"

// Plates number limit
#define PLATES_NBR_LIMIT            100

/*---------------------------------------------------------------*/
/*-------------------  csv files configuration ------------------*/
/*---------------------------------------------------------------*/

// csv files colonnes number
#define BATCH_COL_LENGTH            6
#define SOLUTION_COL_LENGTH         9
#define DEFECTS_COL_LENGTH          6

// Optimization parameters raws number
#define OPT_PARAM_RAWS_LENGTH       8

// Batch file colonnes index
#define BATCH_ITEM_ID_COL           0
#define BATCH_LENGTH_ITEM_COL       1
#define BATCH_WIDTH_ITEM_COL        2
#define BATCH_STACK_COL             3
#define BATCH_SEQ_COL               4

// Solution file colonnes index
#define SOLUTION_PLATE_ID_COL       0
#define SOLUTION_NODE_ID_COL        1
#define SOLUTION_X_COL              2
#define SOLUTION_Y_COL              3
#define SOLUTION_WIDTH_COL          4
#define SOLUTION_HEIGHT_COL         5
#define SOLUTION_TYPE_COL           6
#define SOLUTION_CUT_COL            7
#define SOLUTION_PARENT_COL         8

// Defects file colonnes index
#define DEFECTS_DEFECT_ID_COL       0
#define DEFECTS_PLATE_ID_COL        1
#define DEFECTS_X_COL               2
#define DEFECTS_Y_COL               3
#define DEFECTS_WIDTH_COL           4
#define DEFECTS_HEIGHT_COL          5

// Optimization parameters file raws index
#define OPT_PARAM_NPLATES_RAW             0
#define OPT_PARAM_WIDTHPLATES_RAW        1
#define OPT_PARAM_HEIGHTPLATES_RAW         2
#define OPT_PARAM_MIN1CUT_RAW             3
#define OPT_PARAM_MAX1CUT_RAW             4
#define OPT_PARAM_MIN2CUT_RAW             5
#define OPT_PARAM_MINWASTE_RAW       6

/*---------------------------------------------------------------*/
/*--------  Solution's possible errors mask and offset  --------*/
/*---------------------------------------------------------------*/

// Mask and Offset for errors found while parsing solution file.
#define PARSE_ERROR_MASK                    1
#define PARSE_ERROR_OFFSET                  0

// Mask and Offset for errors for item production constraint verification.
#define ITEM_PRODUCTION_ERROR_MASK          2
#define ITEM_PRODUCTION_ERROR_OFFSET        1

// Mask and Offset for errors for defects superposing constraint verification.
#define DEFECTS_SUPERPOSING_ERROR_MASK      4
#define DEFECTS_SUPERPOSING_ERROR_OFFSET    2

// Mask and Offset for errors for items dimensions constraint verification.
#define DIMENSIONS_ERROR_MASK               8
#define DIMENSIONS_ERROR_OFFSET             3

// Mask and Offset for errors for identity constraint verification.
#define IDENTITY_ERROR_MASK                 16
#define IDENTITY_ERROR_OFFSET               4

// Mask and Offset for errors for sequence constraint verification.
#define SEQUENCE_ERROR_MASK                 32
#define SEQUENCE_ERROR_OFFSET               5

#endif // MAIN_H_INCLUDED
