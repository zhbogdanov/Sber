# Библиотека metrics для расчета различных метрик качества #

## Примеры использования ##

```python
    from metric import Metrics, Error
```

Metrics - класс, в котором реализованы методы для расчета метрик 
Error - класс ошибок 

```python
    # Параметры:
    # y_true  - истинные метки классов
    # y_pred  - предсказанные метки
    # y_probs - предсказанные вероятности меток 
    # task='Classification' - тип задачи (Classification или Regression)
    metr = Metrics(y_true, y_pred, y_probs, task='Classification')
```

### Задача бинарной классификации ###

metr = Metrics(y_true, y_pred, y_probs)

#### Accuracy score ####

```python
    metr.accuracy_score()
```

#### Precision score ####

```python
    metr.precision_score()
```

#### Recall score ####

```python
    metr.recall_score()
```

#### Fbeta score ####

```python
    # Параметры:
    # beta=1 - коэффициент beta
    metr.fbeta_score()
```

#### Log loss ####

```python
    # Параметры:
    # eps=1e-15 - если веростность p = 0 или p = 1, то p = max(eps, min(1 - eps, p))
    metr.log_loss()
```

#### ROC AUC score ####

```python
    metr.roc_auc_score()
```

### Задача регрессии ###

metr = Metrics(y_true, y_pred, y_probs, task='Regression')

#### MAE ####

```python
    metr_reg.mean_absolute_error()
```

#### MSE ####

```python
    # Параметры:
    # squared=True - выбор между MSE RMSE, по умолчанию MSE
    metr_reg.mean_squared_error()
```

#### RMSE ####

```python
    metr_reg.mean_squared_error(squared=False)
```

#### R2 score ####

```python
    metr_reg.r2_score()
```
