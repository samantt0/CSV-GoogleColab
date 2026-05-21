import numpy as np
import warnings

#    ЧАСТЬ 1
#загрузка
data_dtype = np.dtype([('ts', 'i4'), ('node_id', 'i2'), ('qerr', 'f4'), ('coh', 'f4'), ('gates', 'i2'), ('fid', 'f4')])
data = np.genfromtxt('data_variant_57.csv', delimiter=',', dtype=data_dtype)

#основная инф-я
row_count = data.shape[0]
volume_bytes = data.nbytes
volume_mb = volume_bytes / (1024 * 1024)

print(f"Количество строк: {row_count}")
print(f"Объем памяти: {volume_bytes} байт ({volume_mb:.2f} МБ)")

#NaN и Inf
fields = ['qerr', 'coh', 'fid']
total_values = 0
total_nan_inf = 0

for field in fields:
    field_data = data[field]
    nan_count = np.sum(np.isnan(field_data))
    inf_count = np.sum(np.isinf(field_data))
    total = nan_count + inf_count
    total_values += len(field_data)
    total_nan_inf += total

    percentage = (total / len(field_data)) * 100
    print(f"  {field}: NaN={nan_count}, Inf={inf_count}, всего={total} ({percentage:.2f}%)")

overall_percentage = (total_nan_inf / total_values) * 100
print(f"\nОбщая доля NaN/Inf: {overall_percentage:.2f}%")

if overall_percentage > 3:
    warnings.warn(f"ВНИМАНИЕ: Доля NaN/Inf ({overall_percentage:.2f}%) превышает 3%")

#сохранение
np.save('data.npy', data)

#    ЧАСТЬ 2
#аномалии
data_clean = data.copy()
mask_qerr_out = (data['qerr'] < 0) | (data['qerr'] > 1)
mask_coh_neg = data['coh'] < 0
mask_fid_out = (data['fid'] < 0) | (data['fid'] > 1)
mask_nan_inf = np.isnan(data['qerr']) | np.isinf(data['qerr']) | \
               np.isnan(data['coh']) | np.isinf(data['coh']) | \
               np.isnan(data['fid']) | np.isinf(data['fid'])

mask_anomalies = mask_qerr_out | mask_coh_neg | mask_fid_out | mask_nan_inf
anomaly_count = np.sum(mask_anomalies)
anomaly_percentage = (anomaly_count / row_count) * 100
print(f"Найдено аномалий:")
print(f"--  qerr вне [0,1]: {np.sum(mask_qerr_out)}")
print(f"--  coh < 0: {np.sum(mask_coh_neg)}")
print(f"--  fid вне [0,1]: {np.sum(mask_fid_out)}")
print(f"--  nan, inf: {np.sum(mask_nan_inf)}")
print(f"--  ВСЕГО: {anomaly_count} ({anomaly_percentage:.2f}%)")

#очистка
for field in fields:
    mask_invalid = np.isnan(data_clean[field]) | np.isinf(data_clean[field])
    if np.sum(mask_invalid) > 0:
        valid_data = data_clean[field][~mask_invalid]
        if len(valid_data) > 0:
            median_value = np.median(valid_data)
            data_clean[field] = np.where(mask_invalid, median_value, data_clean[field])

data_clean['qerr'] = np.clip(data_clean['qerr'], 0, 1)
data_clean['fid'] = np.clip(data_clean['fid'], 0, 1)
data_clean['coh'] = np.where(data_clean['coh'] < 0, 0, data_clean['coh'])
print(f"Данные очищены")

#    ЧАСТЬ 3
#группировка
unique_nodes, inverse_indices = np.unique(data_clean['node_id'], return_inverse=True)
n_groups = len(unique_nodes)
print(f"Количество групп: {n_groups}")

group_stats = []
for i, node in enumerate(unique_nodes):
    mask_group = data_clean['node_id'] == node
    group_size = np.sum(mask_group)

    coh_group = data_clean['coh'][mask_group]
    gates_group = data_clean['gates'][mask_group]

    mean_coh = np.mean(coh_group)
    std_coh = np.std(coh_group)
    max_gates = np.max(gates_group)

    group_stats.append({
        'node_id': node,
        'size': group_size,
        'mean_coh': mean_coh,
        'std_coh': std_coh,
        'max_gates': max_gates
    })

    print(f"  Node {node}: размер={group_size}, mean(coh)={mean_coh:.4f}, "
          f"std(coh)={std_coh:.4f}, max(gates)={max_gates}")

#нормализация
data_normalized = data_clean.copy()
coh_normalized = np.zeros_like(data_clean['coh'])

epsilon = 1e-8

for i, node in enumerate(unique_nodes):
    mask_group = data_clean['node_id'] == node
    coh_group = data_clean['coh'][mask_group]

    mean_coh = np.mean(coh_group)
    std_coh = np.std(coh_group)

    #Z-score нормализация с защитой от деления на ноль
    coh_normalized[mask_group] = (coh_group - mean_coh) / (std_coh + epsilon)

data_normalized['coh'] = coh_normalized
np.save('data_normalized.npy', data_normalized)

#    ЧАСТЬ 4
#скользящее окно
k = 40

coh_cumsum = np.cumsum(data_clean['coh'])
coh_cumsum = np.insert(coh_cumsum, 0, 0)
coh_moving_avg = (coh_cumsum[k:] - coh_cumsum[:-k]) / k

#дополнение первых k-1 значений
coh_moving_avg_full = np.pad(coh_moving_avg, (k-1, 0), mode='edge')

print(f"Размер окна: k={k}")
print(f"Длина исходного массива coh: {len(data_clean['coh'])}")
print(f"Длина скользящего среднего: {len(coh_moving_avg_full)}")

#вычисление разницы fid между соседними записями
fid_diff = np.diff(data_clean['fid'])
fid_diff_full = np.concatenate([[0], fid_diff])

#создание нового признака
#признак: категория изменения (рост/падение/стабильность)
fid_change_category = np.zeros(row_count, dtype='i1')  # -1: падение, 0: стабильно, 1: рост
threshold = 0.001  #порог для определения "стабильности"

fid_change_category = np.where(fid_diff_full > threshold, 1, fid_change_category)
fid_change_category = np.where(fid_diff_full < -threshold, -1, fid_change_category)

print(f"\nСтатистика изменения fid:")
print(f"  Рост: {np.sum(fid_change_category == 1)}")
print(f"  Стабильно: {np.sum(fid_change_category == 0)}")
print(f"  Падение: {np.sum(fid_change_category == -1)}")

data_with_features_dtype = np.dtype([
    ('ts', 'i4'),
    ('node_id', 'i2'),
    ('qerr', 'f4'),
    ('coh', 'f4'),
    ('coh_ma', 'f4'),
    ('gates', 'u2'),
    ('fid', 'f4'),
    ('fid_diff', 'f4'),
    ('fid_change', 'i1')
])

data_with_features = np.zeros(row_count, dtype=data_with_features_dtype)
for field in ['ts', 'node_id', 'qerr', 'coh', 'gates', 'fid']:
    data_with_features[field] = data_clean[field]
data_with_features['coh_ma'] = coh_moving_avg_full
data_with_features['fid_diff'] = fid_diff_full
data_with_features['fid_change'] = fid_change_category

#    ЧАСТЬ 5
#эффективность узла = fid / (qerr + epsilon)
#высокая точность при низкой ошибке = хорошая эффективность
efficiency = data_clean['fid'] / (data_clean['qerr'] + epsilon)
efficiency = np.where(np.isinf(efficiency) | np.isnan(efficiency),
                      np.median(efficiency[np.isfinite(efficiency)]),
                      efficiency)


#производительность = gates / (coh + epsilon)
#количество операций на единицу времени когерентности
performance = data_clean['gates'].astype('f4') / (data_clean['coh'] + epsilon)
performance = np.where(np.isinf(performance) | np.isnan(performance),
                       np.median(performance[np.isfinite(performance)]),
                       performance)


#обновление структуры данных
data_engineered_dtype = np.dtype([
    ('ts', 'i4'),
    ('node_id', 'i2'),
    ('qerr', 'f4'),
    ('coh', 'f4'),
    ('coh_ma', 'f4'),
    ('gates', 'u2'),
    ('fid', 'f4'),
    ('fid_diff', 'f4'),
    ('fid_change', 'i1'),
    ('efficiency', 'f4'),
    ('performance', 'f4')
])

data_engineered = np.zeros(row_count, dtype=data_engineered_dtype)
for field in data_with_features.dtype.names:
    data_engineered[field] = data_with_features[field]
data_engineered['efficiency'] = efficiency
data_engineered['performance'] = performance

#    ЧАСТЬ 6
#условие: высокая точность (fid > 0.9) и низкая ошибка (qerr < 0.1)
condition_mask = (data_clean['fid'] > 0.9) & (data_clean['qerr'] < 0.1)

print(f"Записей, удовлетворяющих условию: {np.sum(condition_mask)}")

#условная агрегация по группам
conditional_stats = np.zeros((n_groups, 4))  # [node_id, mean, median, p90]

for i, node in enumerate(unique_nodes):
    mask_group = data_clean['node_id'] == node
    combined_mask = mask_group & condition_mask

    if np.sum(combined_mask) > 0:
        coh_conditional = data_clean['coh'][combined_mask]
        conditional_stats[i] = [
            node,
            np.mean(coh_conditional),
            np.median(coh_conditional),
            np.percentile(coh_conditional, 90)
        ]
    else:
        conditional_stats[i] = [node, 0, 0, 0]

    print(f"  Node {node}: mean={conditional_stats[i,1]:.4f}, "
          f"median={conditional_stats[i,2]:.4f}, p90={conditional_stats[i,3]:.4f} "
          f"(N={np.sum(combined_mask)})")

#    ЧАСТЬ 7
#создание лагового признака для fid
fid_lag1 = np.roll(data_clean['fid'], 1)
fid_lag1[0] = data_clean['fid'][0]

#разница между текущим и предыдущим значением
fid_delta = data_clean['fid'] - fid_lag1
fid_delta[0] = 0  #первый элемент без предыдущего

#анализ изменений
mask_increased = fid_delta > 0
mask_decreased = fid_delta < 0
mask_unchanged = fid_delta == 0

count_increased = np.sum(mask_increased)
count_decreased = np.sum(mask_decreased)
count_unchanged = np.sum(mask_unchanged)

pct_increased = (count_increased / row_count) * 100
pct_decreased = (count_decreased / row_count) * 100
pct_unchanged = (count_unchanged / row_count) * 100

print(f"Выросло: {count_increased} ({pct_increased:.2f}%)")
print(f"Упало: {count_decreased} ({pct_decreased:.2f}%)")
print(f"Не изменилось: {count_unchanged} ({pct_unchanged:.2f}%)")

#распределение знаков
signs = np.sign(fid_delta).astype(int) + 1
sign_counts = np.bincount(signs, minlength=3)

print(f"\n")
print(f"Отрицательные: {sign_counts[0]}")
print(f"Нулевые: {sign_counts[1]}")
print(f"Положительные: {sign_counts[2]}")

#    ЧАСТЬ 8
data_iqr_clean = data_clean.copy()
total_replaced = 0
replacements_by_group = {}

for i, node in enumerate(unique_nodes):
    mask_group = data_iqr_clean['node_id'] == node
    coh_group = data_iqr_clean['coh'][mask_group].copy()

    #вычисление квартилей
    Q1 = np.percentile(coh_group, 25)
    Q3 = np.percentile(coh_group, 75)
    IQR = Q3 - Q1

    #границы
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    #поиск выбросов
    outliers_mask = (coh_group < lower_bound) | (coh_group > upper_bound)
    n_outliers = np.sum(outliers_mask)

    if n_outliers > 0:
        median_group = np.median(coh_group)
        coh_group[outliers_mask] = median_group
        data_iqr_clean['coh'][mask_group] = coh_group
        total_replaced += n_outliers
        replacements_by_group[node] = n_outliers

    print(f"  Node {node}: Q1={Q1:.4f}, Q3={Q3:.4f}, IQR={IQR:.4f}, "
          f"bounds=[{lower_bound:.4f}, {upper_bound:.4f}], заменено={n_outliers}")

pct_replaced = (total_replaced / row_count) * 100
print(f"\nВсего заменено выбросов: {total_replaced} ({pct_replaced:.2f}%)")

#    ЧАСТЬ 9
#правила предметной области:
# 1. точность не может быть ниже ошибки (fid >= qerr)
# 2. При высокой ошибке точность должна быть низкой (qerr > 0.5 и fid < 0.8)
# 3. gates не может быть 0 при высокой точности (fid > 0.95)
# 4. операции требуют когерентности (coh > 0 при gates > 0)

rule1_violated = data_iqr_clean['fid'] < data_iqr_clean['qerr']
rule2_violated = (data_iqr_clean['qerr'] > 0.5) & (data_iqr_clean['fid'] >= 0.8)
rule3_violated = (data_iqr_clean['gates'] == 0) & (data_iqr_clean['fid'] > 0.95)
rule4_violated = (data_iqr_clean['gates'] > 0) & (data_iqr_clean['coh'] == 0)

#составная маска
integrity_violations = rule1_violated | rule2_violated | rule3_violated | rule4_violated

n_violations = np.sum(integrity_violations)
pct_violations = (n_violations / row_count) * 100

print(f"Правило 1 (fid < qerr): {np.sum(rule1_violated)}")
print(f"Правило 2 (qerr>0.5 & fid>=0.8): {np.sum(rule2_violated)}")
print(f"Правило 3 (gates=0 & fid>0.95): {np.sum(rule3_violated)}")
print(f"Правило 4 (gates>0 & coh=0): {np.sum(rule4_violated)}")
print(f"ВСЕГО: {n_violations} ({pct_violations:.2f}%)")

#исправление нарушений
data_integrity_fixed = data_iqr_clean.copy()

#если fid < qerr, установить fid = qerr
data_integrity_fixed['fid'] = np.where(rule1_violated,
                                       data_integrity_fixed['qerr'],
                                       data_integrity_fixed['fid'])

#если qerr > 0.5 и fid >= 0.8, снизить fid до 0.7
data_integrity_fixed['fid'] = np.where(rule2_violated,
                                       0.7,
                                       data_integrity_fixed['fid'])

#если gates = 0 при fid > 0.95, установить gates = 1
data_integrity_fixed['gates'] = np.where(rule3_violated,
                                         1,
                                         data_integrity_fixed['gates'])

#если gates > 0 при coh = 0, установить минимальную когерентность
min_coh = np.min(data_integrity_fixed['coh'][data_integrity_fixed['coh'] > 0])
data_integrity_fixed['coh'] = np.where(rule4_violated,
                                       min_coh,
                                       data_integrity_fixed['coh'])


#    ЧАСТЬ 10
#анализ распределения node_id
unique_nodes_freq, counts = np.unique(data_integrity_fixed['node_id'], return_counts=True)
frequencies = counts / row_count

print(f"Распределение узлов (node_id):")
for node, count, freq in zip(unique_nodes_freq, counts, frequencies):
    status = "РЕДКИЙ" if freq < 0.01 else ""
    print(f"  Node {node}: {count} записей ({freq*100:.2f}%) {status}")

#поиск редких категорий
rare_mask = frequencies < 0.01
rare_nodes = unique_nodes_freq[rare_mask]
n_rare_records = np.sum(counts[rare_mask])

print(f"\nРедкие категории (< 1%): {len(rare_nodes)}")
print(f"Записей в редких категориях: {n_rare_records} ({n_rare_records/row_count*100:.2f}%)")

# Объединение редких категорий в OTHER (код 0)
data_final = data_integrity_fixed.copy()
if len(rare_nodes) > 0:
    for rare_node in rare_nodes:
        data_final['node_id'] = np.where(data_final['node_id'] == rare_node,
                                         0,
                                         data_final['node_id'])
    print(f"Редкие категории объединены в категорию 'OTHER' (код 0)")
else:
    print("Редких категорий не обнаружено")