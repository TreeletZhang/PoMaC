from PIL import Image
ascii_char = list("$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. ")

WIDTH = 120 # 字符画的宽
HEIGHT = 60 # 字符画的高


# 将256灰度映射到70个字符上，也就是RGB值转字符的函数：
def get_char(r, g, b, alpha=256):  # alpha透明度
   if alpha == 0:
       return ' '
   length = len(ascii_char)
   gray = int(0.2126 * r + 0.7152 * g + 0.0722 * b)  # 计算灰度
   unit = (256.0 + 1) / length
   return ascii_char[int(gray / unit)]  # 不同的灰度对应着不同的字符
   # 通过灰度来区分色块


if __name__ == '__main__':
   img = './logo_mini.jpg' # 图片所在位置
   im = Image.open(img)
   im = im.resize((WIDTH, HEIGHT), Image.NEAREST)
   im = im.convert('L')
   # im.show()
   txt = ""
   for i in range(HEIGHT):
       for j in range(WIDTH):
           if im.getpixel((j, i)) > 150:
               txt += ' '
           elif im.getpixel((j, i)) > 100:
               txt += '*'
           else:
               txt += '▓'
           # txt += get_char(*im.getpixel((j, i))) # 获得相应的字符
       txt += '\n'
   print(txt)  # 打印出字符画
   # 将字符画 写入文件中
   with open("logo.txt", 'w') as f:
       f.write(txt)