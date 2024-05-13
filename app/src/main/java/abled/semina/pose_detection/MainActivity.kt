package abled.semina.pose_detection

import abled.semina.pose_detection.ml.AutoModel4
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.view.Surface
import android.view.TextureView
import android.widget.ImageView
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class MainActivity : AppCompatActivity() {

    // 원 그리기 위한 Paint 객체
    val paint = Paint()

    // 이미지 처리를 위한 구성요소
    lateinit var imageProcessor: ImageProcessor
    lateinit var model: AutoModel4

    // 캡처된 이미지를 보유할 Bitmap
    lateinit var bitmap: Bitmap

    // 뷰들
    lateinit var imageView: ImageView
    lateinit var textureView: TextureView

    // 카메라 관련 구성요소
    lateinit var cameraManager: CameraManager
    lateinit var handler: Handler
    lateinit var handlerThread: HandlerThread

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // 필요한 권한 요청
        get_permissions()

        // 이미지 처리 설정
        imageProcessor = ImageProcessor
            .Builder()
            .add(ResizeOp(192,192, ResizeOp.ResizeMethod.BILINEAR)).build()

        model = AutoModel4.newInstance(this)

        // 뷰 및 카메라 구성요소 초기화
        imageView = findViewById(R.id.imageView)
        textureView = findViewById(R.id.textureView)
        cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        handlerThread = HandlerThread("videoThread")
        handlerThread.start()
        handler = Handler(handlerThread.looper)

        // 원 그리기 위한 Paint 색상 설정
        paint.color = Color.YELLOW

        // TextureView 리스너 설정하여 카메라 미리보기 처리
        textureView.surfaceTextureListener = object:TextureView.SurfaceTextureListener{
            override fun onSurfaceTextureAvailable(
                surface: SurfaceTexture,
                width: Int,
                height: Int
            ) {
                // TextureView가 사용 가능할 때 카메라 열기
                open_camera()
            }

            override fun onSurfaceTextureSizeChanged(
                surface: SurfaceTexture,
                width: Int,
                height: Int
            ) {
                // TextureView 크기 변경 처리
            }

            override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
                return false
            }

            override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {
                // TextureView가 업데이트될 때 캡처된 이미지 처리

                // TextureView에서 비트맵 가져오기
                bitmap = textureView.bitmap!!

                // 비트맵을 TensorImage에 로드
                var tensorImage = TensorImage(DataType.UINT8)
                tensorImage.load(bitmap)
                tensorImage = imageProcessor.process(tensorImage)

                // 모델 추론을 위한 입력 준비
                val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 192, 192, 3), DataType.UINT8)
                inputFeature0.loadBuffer(tensorImage.buffer)

                // 모델 추론 실행
                val outputs = model.process(inputFeature0)
                val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray

                // 모델 출력을 기반으로 비트맵에 그리기
                var mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true)
                var canvas = Canvas(mutable)

                var h = bitmap.height
                var w = bitmap.width
                var x = 0

                while(x <= 49){
                    if(outputFeature0.get(x+2) > 0.45){
                        canvas.drawCircle(outputFeature0.get(x+1)*w, outputFeature0.get(x)*h, 10f, paint)
                    }
                    x+=3
                }

                // 수정된 비트맵 표시
                imageView.setImageBitmap(mutable)
            }

        }
    }


    // onDestroy()
    override fun onDestroy() {
        super.onDestroy()

        // 모델 자원 해제
        model.close()

    } // onDestroy()


    @SuppressLint("MissingPermission")
    fun open_camera(){
        // 카메라 열기
        cameraManager.openCamera(cameraManager.cameraIdList[0], object : CameraDevice.StateCallback() {
            override fun onOpened(p0: CameraDevice) {
                // 캡처 요청 생성
                var captureRequest = p0.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                var surface = Surface(textureView.surfaceTexture)
                captureRequest.addTarget(surface)

                // 캡처 세션 생성
                p0.createCaptureSession(listOf(surface), object:CameraCaptureSession.StateCallback(){
                    override fun onConfigured(p0: CameraCaptureSession) {
                        // 프레임 캡처 시작
                        p0.setRepeatingRequest(captureRequest.build(),null,null)
                    }

                    override fun onConfigureFailed(p0: CameraCaptureSession) {
                        // 캡처 세션 구성 실패 처리
                    }

                }, handler)
            }

            override fun onDisconnected(camera: CameraDevice) {
                // 카메라 연결 해제 처리
            }

            override fun onError(camera: CameraDevice, error: Int) {
                // 카메라 오류 처리
            }

        }, handler)
    }

    fun get_permissions(){
        // 카메라 권한 확인 및 요청
        if(checkSelfPermission(android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED){
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA),101)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        // 카메라 권한이 부여되지 않은 경우 다시 요청
        if(grantResults[0] != PackageManager.PERMISSION_GRANTED) get_permissions()
    }
}