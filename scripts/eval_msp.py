import atom3d.util.results as res
import atom3d.util.metrics as met
import sys

checkpoint = sys.argv[1]

# Define the training run 
name = f'logs/msp_fixed_{checkpoint}/msp'
print(name)

# Load training results
rloader = res.ResultsGNN(name, reps=[0, 1, 2])
results = rloader.get_all_predictions()

y_true = results['rep0']['targets']['test']
y_pred = results['rep0']['predict']['test']

# Calculate and print results
summary = met.evaluate_average(results, metric = met.auroc, verbose = False)
print('Test AUROC: %6.3f \pm %6.3f' % summary[2])
summary = met.evaluate_average(results, metric = met.auprc, verbose = False)
print('Test AUPRC: %6.3f \pm %6.3f' % summary[2])
