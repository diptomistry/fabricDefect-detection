git clone https://github.com/amazon-science/patchcore-inspection.git
cd patchcore-inspection
pip install .


export KMP_DUPLICATE_LIB_OK=TRUE
python3 patchcore_run.py

KMP_DUPLICATE_LIB_OK=TRUE python3 fabric_defect_detector.py 

1. Test a defect image:
KMP_DUPLICATE_LIB_OK=TRUE python test_single_image.py defect/IMG_0109.jpg
2. Test a normal image:
KMP_DUPLICATE_LIB_OK=TRUE python test_single_image.py noDefect/IMG_0125.jpg
