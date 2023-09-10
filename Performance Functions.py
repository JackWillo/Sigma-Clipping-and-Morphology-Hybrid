def perf(image, mask):
  TP = np.sum((mask == 1) & (image == 1))
  FP = np.sum((mask == 1) & (image == 0))
  FN = np.sum((mask == 0) & (image == 1))
  TN = np.sum((mask == 0) & (image == 0))

  Acc = (TP + TN) / (TP + FP + FN + TN)

  if (TP + FP) == 0:
      Prec = 0  # or np.nan if you want it to be NaN
  else:
      Prec = TP / (TP + FP)


  Recall = TP / (TP + FN)

  if (Prec + Recall) == 0:
    F1 = 0
  else:
    F1 = (2*Prec*Recall)/(Prec + Recall)

  return Acc, TP, FP, FN, TN, Prec, Recall, F1

def perf2(image, mask):
  TP = np.sum((mask == 1) & (image == 1))
  FP = np.sum((mask == 1) & (image == 0))
  FN = np.sum((mask == 0) & (image == 1))
  TN = np.sum((mask == 0) & (image == 0))


  Acc = (TP + TN) / (TP + FP + FN + TN)
  Prec = TP / (TP + FP)
  Recall = TP / (TP + FN)
  F1 = (2*Prec*Recall)/(Prec + Recall)
  Spec =  TN / (TN + FP)
  Bal = 0.5 * (Recall + Spec)

  return(Spec, Bal)