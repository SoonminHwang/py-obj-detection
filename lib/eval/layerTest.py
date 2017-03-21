layer {
  name: 'input-data'
  type: 'Input'
  top: 'image'  
  input_param { 
    shape { dim: 1      dim: 3      dim: 375      dim: 1242 }    
  }
}

layer {
  type: 'Python'
  name: 'mAP'  
  top: 'mAP_Car'
  top: 'mAP_Ped'
  top: 'mAP_Cyc'  
  #bottom: 'bbox_list'
  #bottom: 'gt_boxes'
  bottom: 'image'
  #bottom: 'im_info'
  python_param {
      module: 'eval.detectnet'
      layer: 'EvalLayer'
      # parameters (default)
      #   - DIFFICULTY = {'easy': 0, 'moderate': 1, 'hard': 2}
      #   - MIN_HEIGHT = (40, 25, 25)
      #   - MAX_OCCLUSION = (0, 1, 2)
      #   - MAX_TRANCATION = (0.15, 0.3, 0.5)
      #   - CLASSES = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
      #   - NEIGHBOR_CLASSES = {'Car': ['Van'], 'Pedestrian': ['Person_sitting'], 'Cyclist': []]}
      #   - MIN_OVERLAP = (0.7, 0.5, 0.5)
      #   - N_SAMPLE_PTS = 41
      # Ex) param_str : "{MIN_OVERLAP: (0.7, 0.5, 0.5)}
      param_str : "{}"
  }
  include: { phase: TEST}
}