import cv2
import numpy as np
import datetime

moto = cv2.imread('chess/site_1.png')
hikaku = cv2.imread('chess/site_2.png')

#画像サイズ（縦、横）の取り込み
h_1, w_1, _=moto.shape
h_2, w_2, _=hikaku.shape

#縦、横サイズ、それぞれ小さい方を見つける
if h_1>h_2:
    h_min=h_2
else:
    h_min=h_1
if w_1>w_2:
    w_min=w_2
else:
    w_min=w_1

#同じサイズにトリミング
moto=moto[0:h_min, 0:w_min]
hikaku=hikaku[0:h_min, 0:w_min]

#特徴量抽出アルゴリズム ORB (Oriented FAST and Rotated BRIEF)
detector = cv2.ORB_create()

#画像のぼかしを追加
moto = cv2.GaussianBlur(moto, (5,5),0)
hikaku = cv2.GaussianBlur(hikaku, (5, 5), 0)

#それぞれの画像の特徴点と特徴量
moto_kp, moto_desc = detector.detectAndCompute(moto, None)
hikaku_kp, hikaku_desc = detector.detectAndCompute(hikaku, None)

#マッチングの準備
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#マッチング
matches = matcher.match(moto_desc, hikaku_desc)

#画像同士の対応する座標リストの作成
coordinate_moto=[]
coordinate_hikaku=[]

#良いマッチングのものだけ抜き出す
for i in range(len(matches)):
    if matches[i].distance<20:
        coordinate_moto.append([moto_kp[matches[i].queryIdx].pt[0],moto_kp[matches[i].queryIdx].pt[1]])
        coordinate_hikaku.append([hikaku_kp[matches[i].trainIdx].pt[0],hikaku_kp[matches[i].trainIdx].pt[1]])

#numpyアレイに変更
coordinate_moto=np.float32(coordinate_moto)
coordinate_hikaku=np.float32(coordinate_hikaku)

#グレー画像の輝度の差で違いを見つけるため、グレーに変換
moto_gray=cv2.cvtColor(moto, cv2.COLOR_BGR2GRAY)
hikaku_gray=cv2.cvtColor(hikaku, cv2.COLOR_BGR2GRAY)

#マッチングした座標で変換行列を作成
matrix, _ =cv2.findHomography(coordinate_hikaku, coordinate_moto, cv2.RANSAC) 
#比較画像を元画像に合わせるための射影変換
hikaku = cv2.warpPerspective(hikaku, matrix, (w_min, h_min)) 



#ここで重ね合わせ画像も作っておく
moto_and_hikaku=cv2.addWeighted(src1=moto, alpha=0.5, src2=hikaku, beta=0.5, gamma=0)

#調整用パラメータ
th_value=3  #輝度の差のしきい値
kernel_value=3  #ノイズを消すピクセルサイズ

# #画像の比較、輝度の差がしきい値以上のものは、白く、それ以外のところは黒にする
# img_diff=cv2.absdiff(moto_gray, hikaku_gray)
# _, img_diff=cv2.threshold(img_diff, th_value, 255, cv2.THRESH_BINARY)
# kernel=np.ones((kernel_value,kernel_value), dtype=np.uint8)
# img_diff=cv2.morphologyEx(img_diff, cv2.MORPH_OPEN, kernel)

# #白い部分の輪郭を検出
# contours, _=cv2.findContours(img_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# #輪郭を赤枠の四角に変換
# rectangle_img=np.zeros((h_min,w_min,3),np.uint8) #真っ黒の画像を作る
# for ct in contours:
#     x, y, ww, hh= cv2.boundingRect(ct) #輪郭を位置を抽出
#     rectangle_img=cv2.rectangle(rectangle_img, (x-5,y-5),(x+ww+5,y+hh+5),(0, 0, 255), 2)

# #画像に赤枠を重ね合わせ
# img_result = cv2.addWeighted(src1=rectangle_img,alpha=1,src2=moto_and_hikaku,beta=1,gamma=0)

#画像を確認
cv2.imshow('img_result', hikaku)
cv2.waitKey()
cv2.destroyAllWindows()

#結果を保存
# タイムスタンプを取得
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# 画像の保存
output_filename = f'output/chess_site_diff_{timestamp}.png'
cv2.imwrite(output_filename, img_result)

