Usage:

```
make
./fluid
```

mp4 generated using

```
ffmpeg -framerate 24 -i fluid_video/frame_%d.png -c:v libx264 -pix_fmt yuv420p fluid_animation.mp4
```
