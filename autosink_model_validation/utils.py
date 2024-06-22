from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import wandb
def val_accuracy(trainer, caseset):
  outputs = trainer.predict(caseset)
  return outputs


def artifact_aliases_append():
  project = "AIsink-resnet50"
  entity = "zzangsu"
  artifact_name = "model-ai-sink-run:latest"  # 예: "model:latest"

  # API 인스턴스 생성
  api = wandb.Api()

  # 아티팩트 불러오기
  artifact = api.artifact(f"{entity}/{project}/{artifact_name}")

  # 별칭 추가
  artifact.aliases.append("success")
  artifact.save()

  print(f"Added alias 'success' to artifact {artifact_name}")


def check_model(outputs_list, thres=0.95):
  # test_acc = outputs.metrics['test_accuracy']
  #outputs_list:[outputs.metrics['test_accuracy'],outputs.metrics['test_accuracy'],,,]
  if all(value >= thres for value in outputs_list):
    print("success")  # 출력: success
    artifact_aliases_append()
  else:
    print("not success")

def confusionmatrix(outputs, eval_dataset):
  y_true = outputs.label_ids
  y_pred = outputs.predictions.argmax(1)

  labels = eval_dataset.features['label'].names
  cm = confusion_matrix(y_true, y_pred)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
  disp.plot(xticks_rotation=45)
  return disp