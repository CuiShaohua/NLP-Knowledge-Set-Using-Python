from bs4 import BeautifulSoup
import requests
import lxml
import re
import os
import json
# 地铁message
line_rails = '''
body > div > div.content > div.sub_map > div.line_content > div:nth-child(1) > div
body > div > div.content > div.sub_map > div.line_content > div:nth-child(25) > div
body > div > div.content > div.sub_map > div.line_content > div:nth-child(44) > div
body > div > div.content > div.sub_map > div.line_content > div:nth-child(69) > div
body > div > div.content > div.sub_map > div.line_content > div:nth-child(93) > div
body > div > div.content > div.sub_map > div.line_content > div:nth-child(128) > div
body > div > div.content > div.sub_map > div.line_content > div:nth-child(150) > div
body > div > div.content > div.sub_map > div.line_content > div:nth-child(183) > div
body > div > div.content > div.sub_map > div.line_content > div:nth-child(197) > div
body > div > div.content > div.sub_map > div.line_content > div:nth-child(243) > div
body > div > div.content > div.sub_map > div.line_content > div:nth-child(260) > div
body > div > div.content > div.sub_map > div.line_content > div:nth-child(291) > div
body > div > div.content > div.sub_map > div.line_content > div:nth-child(312) > div
body > div > div.content > div.sub_map > div.line_content > div:nth-child(323) > div
body > div > div.content > div.sub_map > div.line_content > div:nth-child(337) > div
body > div > div.content > div.sub_map > div.line_content > div:nth-child(350) > div
body > div > div.content > div.sub_map > div.line_content > div:nth-child(365) > div
body > div > div.content > div.sub_map > div.line_content > div:nth-child(377) > div
body > div > div.content > div.sub_map > div.line_content > div:nth-child(382) > div
body > div > div.content > div.sub_map > div.line_content > div:nth-child(395) > div
'''
user_agent ='Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36'
headers = {'User_Agent': user_agent}
sessions = requests.session()
sessions.keep_alive = False
url = 'https://www.bjsubway.com/station/xltcx/'
#url = 'https://www.bjsubway.com/e/action/ListInfo/?classid=39&ph=1'
sessions.headers = headers
content = sessions.get(url,allow_redirects=True,verify = False)
content.encoding = 'gbk'

soup = BeautifulSoup(content.text, 'lxml')

rails_num =[]
row_num = []
line_pattern = r'(\d+.^A\w+|\w+线)'  # 匹配地铁线编号
line_row_num = r'(\d+)'
station_re = r'>(\w+)'

sum_count = list(range(1,403))

def Arr_split(a, b):
    num_arr1 = []
    num_arr2 = []
    num_arr3 = []
    for i in a:
        num_arr1.append(i)
        # print(num_arr1)
        if num_arr1[-1] in b:
            num_arr1.pop()
            # b.pop(0)
            num_arr2 += [num_arr1.copy()]  # 可变对象赋值果然是个坑
            # print(num_arr2)
            num_arr1.clear()
            b.pop(0)
        elif not b:
            num_arr2 += [a[a.index(i):]]
            return num_arr2


def rail_meg():
    railways_meg = {}
    rail_lines_msg = list()
    for line in line_rails.split('\n'):
        if not line: continue
        # print(str(line))
        row_num.append(int(re.findall(line_row_num, str(line))[0]))
        #print(row_num)
        sets = str(soup.select(line)[0])
        rail_lines_msg.append(re.findall(line_pattern, sets))
    #return rail_lines_msg
    # 切片，除去每个地铁的线路信息
    #return rail_lines_msg
    per_rail_stations_num_sets = Arr_split(sum_count, row_num)
    #return  per_rail_stations_num_sets

    station = []
    stations = []
    for per_rail_station__num in per_rail_stations_num_sets:
        if not per_rail_station__num: continue
        #print(per_rail_station__num)
        for rail_station__num in per_rail_station__num:
            try:
                rail_station_with_tag = str(soup.select('body > div > div.content > div.sub_map > div.line_content > div:nth-child(' +
                                                    str(rail_station__num) + ') > a')[0])
            except IndexError:
                rail_station_with_tag = str(
                    soup.select('body > div > div.content > div.sub_map > div.line_content > div:nth-child(' +
                                str(rail_station__num) + ')')[0])

            station.append(re.findall(station_re, rail_station_with_tag)[0])
        stations += [station.copy()]
        station.clear()

    for i in range(len(rail_lines_msg)):
          railways_meg[rail_lines_msg[i][0]] = stations[i]

    return railways_meg  # 地铁站和线路的对应信息


def check_local_subway_file():
    for root, dirs, files in os.walk(".", topdown=False):
        if 'subway.txt' not in files:  # 这句到底是不是查当前目录
            # print(False)
            json_dict = json.dumps(rail_meg())
            f = open('subway.txt', 'w')
            f.write(json_dict)
            subway_data = json.loads(subway_dict)
            f.close()
        else:
            f = open('subway.txt', 'r')
            subway_dict = f.read()
            subway_data = json.loads(subway_dict)

    return subway_data

def coords(address):
    url = 'https://restapi.amap.com/v3/geocode/geo'   # 输入API问号前固定不变的部分
    params = { 'key': '14bf75d19136fd7ff1043e36b47eb26a',
               'address': address}                    # 将两个参数放入字典
    res = requests.get(url, params)
    jd =  json.loads(res.text)
    return jd['geocodes'][0]['location']


def get_lat_and_long(subway_data):
    station_location = {}

    for subway_nums, subway_stations in subway_data.items():
        for station in subway_stations:
            station_address = '北京' + station + '地铁站'
            long, lat = coords(station_address).split(',')
            long, lat = float(long), float(lat)
            station_location[station] = (long, lat)
    return station_location


def main():

    # 1如果本地文件没有subway的信息，就主动去爬取网页获得
    subway_data = check_local_subway_file()
    # 2 获取整个地铁站的经纬度信息
    print(get_lat_and_long(subway_data))


if __name__ == '__main__':
    main()
