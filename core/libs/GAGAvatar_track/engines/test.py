frame = frames_data[0]
frame_numpy = frame.permute(1, 2, 0).cpu().numpy()
frame_numpy = (frame_numpy*255).astype(np.uint8)
frame_numpy = cv2.cvtColor(255-frame_numpy, cv2.COLOR_RGB2BGR)
for idx, (x, y) in enumerate(lmks):
    cv2.circle(frame_numpy, (int(x.item()), int(y.item())), radius=1, color=(0, 0, 255), thickness=2)
    cv2.putText(frame_numpy, str(idx), (int(x.item()) +5, int(y.item()) +5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(255, 255, 255), thickness=1)
output_tensor = torch.from_numpy(cv2.cvtColor(frame_numpy, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
torchvision.io.write_png(output_tensor, "output_image.png")