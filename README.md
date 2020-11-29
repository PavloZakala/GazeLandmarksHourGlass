# Gaze Landmarks Hour Glass

[Base paper](https://arxiv.org/pdf/1805.04771.pdf)

## 1. Постановка задачі

Дано картинку на які зображено людське око. Знайти розташування на даній картинці наступний клюх ключових точок:

![Eye with landmarks](https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/image_with_landmarks.png?raw=true)

* 8 точок, що описують краї зрачка (ціанові точки)
* 7 точок, що описують краї вій (сині точки)
* 1 точка, що описує центр зіниця (рожеві точки)
* 1 точка, що описує центр очного яблука (зелена точка)

## 1. Опис підхіду

В якості бази для тренування моделі було взято [Hourglass](https://arxiv.org/pdf/1603.06937.pdf), що складається із серії encoder та decoder шарів:

![Hourglass model for eyelandmarks detection](https://github.com/ZPavlo/GazeLandmarksHourGlass/blob/main/sources/hourglass_model.png?raw=true)

На вхід модель отримує кольорову картинку ока формату BGR розміром 128х96. На виході модель повертає теплову мапу 64x48, де більше значення означає більшу вірогідність того що в точці знаходиться ключова точка. 

В якості лосс фукції було взято MSE loss. Функція порівнює вихід кожного з decoder шарів з цільовою тепловою мапою.


## 2. Датасет 


## 3. Реалізація

## 4. Тестування
