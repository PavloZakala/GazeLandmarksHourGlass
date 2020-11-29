# Gaze Landmarks Hour Glass

[Base paper](https://arxiv.org/pdf/1805.04771.pdf)

## 1. Постановка задачі

Дано картинку на які зображено людське око. Знайти розташування на даній картинці наступний клюх ключових точок:

![Eye with landmarks](https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/image_with_landmarks.png?raw=true)

* 8 точок, що описують краї зрачка (ціанові точки)
* 7 точок, що описують краї вій (сині точки)
* 1 точка, що описує центр зіниця (рожеві точки)
* 1 точка, що описує центр очного яблука (зелена точка)

## 2. Опис підхіду

В якості бази для тренування моделі було взято [Hourglass](https://arxiv.org/pdf/1603.06937.pdf), що складається із серії encoder та decoder шарів:

![Hourglass model for eyelandmarks detection](https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/hourglass_model.png?raw=true)

На вхід модель отримує кольорову картинку ока формату BGR розміром 128х72. На виході модель повертає теплову мапу 64x36, де більше значення означає більшу вірогідність того що в точці знаходиться ключова точка. 

В якості лосс фукції було взято MSE loss. Функція порівнює вихід кожного з decoder шарів з цільовою тепловою мапою.


## 3. Датасет 

Дані для тренування було з генеровано штучно за допомогою програми [UnityEyes](https://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/). Проте, для того щоб досягнути високої точності на натуральному датасеті викоистовуючи при цьому штучний для навчання, потрібно реалізувати складну систему аугментації даних. У список функцій для аугментації даних входять:
* scale - Збільшення або зменшення зображення на певний випадковий коефіцієнт;
* gamma_corrected - Грама корекція зображення на випадковий коефіцієнт;
* rotation - Поворот зображення навколо центру на випадковий кут;
* blur - Замилення зображення з випадковою дисперсією функції Гаусса;
* down_up_scale - Зменшення розширення зображення на випадковий коефіцієнт, а після цього повернення розмірів до початкової розмірності;
* add_line - Випадковим чином додати n випадкових ліній. 

Також для покращення процесунавчання було додано коефіцієнт difficult. Чим більше значення difficult тим складнішою є аугментація даних:

<table with difficult progression>

Для кращого навчання моделі ключові точки на картинці було перетворено на формат більш придатний для навчання: набір точок 17х2 перетворено на набір термературних карт розмірністю 17х64x36, де в точці (x<sub>i</sub>, y<sub>i</sub>) знаходиться Гаусівський розподіл, дисперсія якого залежиться від значення difficult.

## 4. Реалізація

Для створення даних для тренування отрібно згенерувати їх з допомогою програми [UnityEyes](https://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/), після цього змініть шлях до згенерованих файлів: в класі EyeLandmarksDataset змініть змінну Path на відповідний шлях. Далі запустити: 

```
> python utils/data_preprocessing.py
```

Тренування моделі було проведено за допомогою модуля PyTorch. Для запуску тренування потрібно виклакати:

```
> python train.py
```
Для тестування:
```
> python test.py
```

Також додано файл Jupyter notebook, який можна запустити з допомогою сервісу Colab. 

Тренування тривало 5 епох, кожна з яких пройшла 1100 бетчів по 16 картинок в одному (batch size). 

## 5. Тестування

Тестування проводилось на чоритьох датасетах: 1 штучно згенерований (UnityEyes) та 3 реальних (CAVE, DIRL, MPII)

### UnityEyes
[source](https://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/)
<table>
  <tr>
    <td>Цільова теплова мапа</td>
    <td>Передбачена теплова мапа</td>
    <td>Цільові ключові точки</td>
    <td>Передбачені ключові точки</td>
   </tr> 
     <tr>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/unityeyes/unityeyes_1_target_heatmap.jpg?raw=true" alt="target_heatmap" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/unityeyes/unityeyes_1_predict_heatmap.jpg?raw=true" alt="predict_heatmap" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/unityeyes/unityeyes_1_target_landmarks.jpg?raw=true" alt="target_landmarks" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/unityeyes/unityeyes_1_predict_landmarks.jpg?raw=true" alt="predict_landmarks" width = 256px height = 144px></td>
  </tr>
     <tr>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/unityeyes/unityeyes_2_target_heatmap.jpg?raw=true" alt="target_heatmap" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/unityeyes/unityeyes_2_predict_heatmap.jpg?raw=true" alt="predict_heatmap" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/unityeyes/unityeyes_2_target_landmarks.jpg?raw=true" alt="target_landmarks" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/unityeyes/unityeyes_2_predict_landmarks.jpg?raw=true" alt="predict_landmarks" width = 256px height = 144px></td>
  </tr>
     <tr>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/unityeyes/unityeyes_3_target_heatmap.jpg?raw=true" alt="target_heatmap" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/unityeyes/unityeyes_3_predict_heatmap.jpg?raw=true" alt="predict_heatmap" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/unityeyes/unityeyes_3_target_landmarks.jpg?raw=true" alt="target_landmarks" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/unityeyes/unityeyes_3_predict_landmarks.jpg?raw=true" alt="predict_landmarks" width = 256px height = 144px></td>
  </tr>
     <tr>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/unityeyes/unityeyes_4_target_heatmap.jpg?raw=true" alt="target_heatmap" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/unityeyes/unityeyes_4_predict_heatmap.jpg?raw=true" alt="predict_heatmap" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/unityeyes/unityeyes_4_target_landmarks.jpg?raw=true" alt="target_landmarks" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/unityeyes/unityeyes_4_predict_landmarks.jpg?raw=true" alt="predict_landmarks" width = 256px height = 144px></td>
  </tr>
     <tr>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/unityeyes/unityeyes_5_target_heatmap.jpg?raw=true" alt="target_heatmap" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/unityeyes/unityeyes_5_predict_heatmap.jpg?raw=true" alt="predict_heatmap" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/unityeyes/unityeyes_5_target_landmarks.jpg?raw=true" alt="target_landmarks" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/unityeyes/unityeyes_5_predict_landmarks.jpg?raw=true" alt="predict_landmarks" width = 256px height = 144px></td>
  </tr>
</table>


### CAVE
[source](https://www.cs.columbia.edu/CAVE/databases/columbia_gaze/)

<table>
  <tr>
    <td>Передбачена теплова мапа</td>
    <td>Передбачені ключові точки</td>
    <td>Передбачена теплова мапа</td>
    <td>Передбачені ключові точки</td>
   </tr> 
   <tr>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/cave/cave_0_heatmap.jpg?raw=true" alt="predict_heatmap" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/cave/cave_0_landmarks.jpg?raw=true" alt="predict_landmarks" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/cave/cave_1_heatmap.jpg?raw=true" alt="predict_heatmap" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/cave/cave_1_landmarks.jpg?raw=true" alt="predict_landmarks" width = 256px height = 144px></td>
  </tr>
     <tr>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/cave/cave_2_heatmap.jpg?raw=true" alt="predict_heatmap" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/cave/cave_2_landmarks.jpg?raw=true" alt="predict_landmarks" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/cave/cave_3_heatmap.jpg?raw=true" alt="predict_heatmap" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/cave/cave_3_landmarks.jpg?raw=true" alt="predict_landmarks" width = 256px height = 144px></td>
  </tr>
     <tr>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/cave/cave_4_heatmap.jpg?raw=true" alt="predict_heatmap" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/cave/cave_4_landmarks.jpg?raw=true" alt="predict_landmarks" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/cave/cave_5_heatmap.jpg?raw=true" alt="predict_heatmap" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/cave/cave_5_landmarks.jpg?raw=true" alt="predict_landmarks" width = 256px height = 144px></td>
  </tr>
     <tr>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/cave/cave_6_heatmap.jpg?raw=true" alt="predict_heatmap" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/cave/cave_6_landmarks.jpg?raw=true" alt="predict_landmarks" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/cave/cave_8_heatmap.jpg?raw=true" alt="predict_heatmap" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/cave/cave_8_landmarks.jpg?raw=true" alt="predict_landmarks" width = 256px height = 144px></td>
  </tr>
</table>


### DIRL

[source](https://sites.google.com/site/chihfanhsuwebsite/dataset)
<table>
  <tr>
    <td>Передбачена теплова мапа</td>
    <td>Передбачені ключові точки</td>
    <td>Передбачена теплова мапа</td>
    <td>Передбачені ключові точки</td>
   </tr> 
   <tr>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/dirl/dirl_0_heatmap.jpg?raw=true" alt="predict_heatmap" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/dirl/dirl_0_landmarks.jpg?raw=true" alt="predict_landmarks" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/dirl/dirl_1_heatmap.jpg?raw=true" alt="predict_heatmap" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/dirl/dirl_1_landmarks.jpg?raw=true" alt="predict_landmarks" width = 256px height = 144px></td>
  </tr>
     <tr>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/dirl/dirl_2_heatmap.jpg?raw=true" alt="predict_heatmap" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/dirl/dirl_2_landmarks.jpg?raw=true" alt="predict_landmarks" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/dirl/dirl_4_heatmap.jpg?raw=true" alt="predict_heatmap" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/dirl/dirl_4_landmarks.jpg?raw=true" alt="predict_landmarks" width = 256px height = 144px></td>
  </tr>
     <tr>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/dirl/dirl_5_heatmap.jpg?raw=true" alt="predict_heatmap" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/dirl/dirl_5_landmarks.jpg?raw=true" alt="predict_landmarks" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/dirl/dirl_6_heatmap.jpg?raw=true" alt="predict_heatmap" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/dirl/dirl_6_landmarks.jpg?raw=true" alt="predict_landmarks" width = 256px height = 144px></td>
  </tr>
     <tr>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/dirl/dirl_8_heatmap.jpg?raw=true" alt="predict_heatmap" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/dirl/dirl_8_landmarks.jpg?raw=true" alt="predict_landmarks" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/dirl/dirl_9_heatmap.jpg?raw=true" alt="predict_heatmap" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/dirl/dirl_9_landmarks.jpg?raw=true" alt="predict_landmarks" width = 256px height = 144px></td>
  </tr>
</table>

### MPII

[source](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild)
<table>
  <tr>
    <td>Передбачена теплова мапа</td>
    <td>Передбачені ключові точки</td>
    <td>Передбачена теплова мапа</td>
    <td>Передбачені ключові точки</td>
   </tr> 
   <tr>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/mpii/mpii_1_heatmap.jpg?raw=true" alt="predict_heatmap" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/mpii/mpii_1_landmarks.jpg?raw=true" alt="predict_landmarks" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/mpii/mpii_3_heatmap.jpg?raw=true" alt="predict_heatmap" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/mpii/mpii_3_landmarks.jpg?raw=true" alt="predict_landmarks" width = 256px height = 144px></td>
  </tr>
     <tr>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/mpii/mpii_4_heatmap.jpg?raw=true" alt="predict_heatmap" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/mpii/mpii_4_landmarks.jpg?raw=true" alt="predict_landmarks" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/mpii/mpii_6_heatmap.jpg?raw=true" alt="predict_heatmap" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/mpii/mpii_6_landmarks.jpg?raw=true" alt="predict_landmarks" width = 256px height = 144px></td>
  </tr>
     <tr>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/mpii/mpii_7_heatmap.jpg?raw=true" alt="predict_heatmap" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/mpii/mpii_7_landmarks.jpg?raw=true" alt="predict_landmarks" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/mpii/mpii_9_heatmap.jpg?raw=true" alt="predict_heatmap" width = 256px height = 144px></td>
      <td><img src="https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/mpii/mpii_9_landmarks.jpg?raw=true" alt="predict_landmarks" width = 256px height = 144px></td>
  </tr>
</table>




