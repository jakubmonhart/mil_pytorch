import numpy as np
import torch

def create_bags(data, instance_labels, pos = 100, neg = 100, max_instances = 7):
   data = np.array(data)
   instance_labels = np.array(instance_labels)
   labels = np.empty(0)
   instances = np.empty(shape = (0, len(data[0])))
   # subbag_id = 0
   instance_id = 0
   # subbag_ids = np.empty(0)
   instance_ids = np.empty(0)

   while pos+neg > 0:
      # Create top bag
      temp_instances = np.empty(shape = (0, len(data[0])))
      is_positive = False
      # n_subbags = np.random.randint(low = 2, high = max_subbags+1)
      temp_instance_id = instance_id
      # temp_subbag_ids = np.empty(0)
      temp_instance_ids = np.empty(0)

      # Choose random labels
      n_instances = np.random.randint(low = 2, high = max_instances+1)
      random_labels = np.random.randint(low = 0, high = 10, size = n_instances)

      # Choose random instances based on labels
      for label in random_labels:
         label_index = np.where(instance_labels == label)[0]
         random_index = np.random.choice(a = label_index, size = 1)
         temp_instances = np.concatenate((temp_instances, data[random_index]), axis = 0)

      # Determine label of sub bag
      if ((7 in random_labels) and (3 not in random_labels)):
         is_positive = True

      # Create ids
      temp_instance_ids = np.concatenate((temp_instance_ids, [temp_instance_id]*n_instances), axis = 0)
      # temp_subbag_ids = np.concatenate((temp_subbag_ids, [subbag_id]*n_instances), axis = 0)
      temp_instance_id += 1

      # Decide, if there is enough of positive or negative top bags
      if (is_positive and pos > 0) or (not is_positive and neg > 0):
         instances = np.concatenate((instances, temp_instances), axis = 0)
         instance_ids = np.concatenate((instance_ids, temp_instance_ids), axis = 0)
         # subbag_ids = np.concatenate((subbag_ids, temp_subbag_ids), axis = 0)
         # subbag_id += 1
         instance_id = temp_instance_id

         if is_positive:
            pos -= 1
            labels = np.concatenate((labels, [1]), axis = 0)
         else:
            neg -= 1
            labels = np.concatenate((labels, [0]))

   return torch.Tensor(instances).float(), torch.Tensor(instance_ids).long(), torch.Tensor(labels).long()
