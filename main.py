from grabscreen import grab_screen
import cv2

for i in range(100):
    screen = grab_screen(region=(0, 0, 1920, 1080))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    screen = cv2.resize(screen, (480, 270))
    cv2.imshow('cv2screen', screen)
    cv2.waitKey(10)
cv2.destroyAllWindows()