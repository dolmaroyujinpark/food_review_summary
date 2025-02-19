from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import csv
from selenium.webdriver.chrome.options import Options

# 헤드리스 모드 설정
chrome_options = Options()
chrome_options.add_argument("--headless")  # 크롬 창을 띄우지 않음

# CSV 파일 초기화
output_file = 'reviews23-2.csv'
with open(output_file, mode='w', newline='', encoding='utf-8-sig') as file:
    writer = csv.writer(file)
    writer.writerow(['Place Name', 'Review'])  # 헤더 작성

url = 'https://map.kakao.com/'
driver = webdriver.Chrome(options=chrome_options)  # 드라이버 경로
driver.get(url)

stations = ["진주 맛집"]


def crawling(page, is_first_page):
    time.sleep(0.2)

    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    store_lists = soup.select('.placelist > .PlaceItem')
    # 검색된 장소 목록 크롤링하기
    for i, place in enumerate(store_lists):
        place_name = place.select('.head_item > .tit_name > .link_name')[0].text  # 가게 이름

        detail_page_xpath = '//*[@id="info.search.place.list"]/li[' + str(i + 1) + ']/div[5]/div[4]/a[1]'

        if page != 1:
            driver.find_element_by_xpath(detail_page_xpath).send_keys(Keys.ENTER)
            driver.switch_to.window(driver.window_handles[-1])  # 상세정보 탭으로 변환
            time.sleep(1)
            print('####', place_name)
            extract_reviews(place_name)
            driver.close()

        driver.switch_to.window(driver.window_handles[0])  # 검색 탭으로 전환

def extract_reviews(place_name):
    """
    리뷰를 크롤링하여 중복을 방지하며 CSV 파일에 저장합니다.
    """
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    review_lists = soup.select('.list_evaluation > li')
    seen_reviews = set()  # 이미 읽은 리뷰를 저장할 집합

    # 초기 리뷰 수집
    for review in review_lists:
        comment = review.select_one('.txt_comment > span')  # 리뷰 텍스트
        if comment:
            review_text = comment.text.strip()
            if review_text and review_text not in seen_reviews:  # 공백 제외 및 중복 방지
                save_to_csv(place_name, review_text)
                seen_reviews.add(review_text)  # 이미 읽은 리뷰로 추가
                print(f"Place: {place_name}, Review: {review_text}")

    # 다음 페이지로 이동하며 추가 리뷰 끝까지 읽기
    next_page = True
    while next_page:
        try:
            next_button = driver.find_element_by_link_text('후기 더보기')  # '더보기' 버튼
            next_button.click()
            time.sleep(1)  # 클릭 후 로딩 대기

            # 추가 리뷰 로드
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            review_lists = soup.select('.list_evaluation > li')

            for review in review_lists:
                comment = review.select_one('.txt_comment > span')  # 리뷰 텍스트
                if comment:
                    review_text = comment.text.strip()
                    if review_text and review_text not in seen_reviews:  # 공백 제외 및 중복 방지
                        save_to_csv(place_name, review_text)
                        seen_reviews.add(review_text)  # 이미 읽은 리뷰로 추가
                        print(f"Place: {place_name}, Review: {review_text}")
        except NoSuchElementException:
            next_page = False  # '후기 더보기' 버튼이 없으면 종료

def save_to_csv(place_name, review):
    # CSV 파일에 가게 이름과 리뷰 저장
    with open(output_file, mode='a', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow([place_name, review])

# 음식점 입력 후 찾기 버튼 클릭
for index, station in enumerate(stations):
    search_area = driver.find_element(By.XPATH, '//*[@id="search.keyword.query"]')  # 검색창

    # 기존 검색어 지우기
    search_area.clear()
    search_area.send_keys(station)
    time.sleep(2)
    driver.find_element(By.XPATH, '//*[@id="search.keyword.submit"]').send_keys(Keys.ENTER)
    time.sleep(2)

    # 장소 버튼 클릭
    driver.find_element(By.XPATH, '//*[@id="info.main.options"]/li[2]/a').send_keys(Keys.ENTER)
    #################### 이 위까지는 고정 ###################

    # 첫 번째 역의 첫 번째 페이지에서만 헤더를 추가하기 위해 index==0 and page==1 조건을 적용
    crawling(1, index == 0 and True)

    try:
        # 장소 더보기 버튼 누르기
        btn = driver.find_element(By.CSS_SELECTOR, '.more')
        driver.execute_script("arguments[0].click();", btn)

        for i in range(2, 5):
            # 페이지 넘기기
            xPath = '//*[@id="info.search.page.no' + str(i) + '"]'
            driver.find_element(By.XPATH, xPath).send_keys(Keys.ENTER)
            time.sleep(1)
            crawling(i, False)

    except:
        print('ERROR!')

    print('**크롤링 완료**')

driver.quit()