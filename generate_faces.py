import yaml				#YAML: 설저파일로딩
from train_gan3d import shape_GAN	#3D 형태 새성 GAN 모델
import os				
from test_gan3d import Test		#GAN 테스트 클래스
import torch				#파이토치 사용하기

from test_texture import test_texture	# 텍스쳐 생성함수
from texture_model.progan_modules import Generator, Discriminator #GAN 기반 텍스쳐 생성기 판별기
from texture_model.assets.texture_processing import add_tex# 
from data.edit_obj import get_mtl
import argparse				#argparse: 머신러닝에서 하이퍼 파라미터 쉽게 넘겨줄 수 있음.



def generate_shapes(num_imgs=1, exp_list=[1,2]):
	"""
 	주어진 개수만큼 얼굴형태 생성후 저장
   	이 함수에서 사용하는 `device` 및 `args` 인자는
    	__main__ 영역에서 argparse로 파싱된 `args`와
    	torch.device로 설정된 `device`를 전역(global)으로 참조합니다.
  	:param num_imgs: 생성 얼굴 샘플개수
   	:param exp_list: 표현 표정 인덱스 리스트
  	"""
	#설정파일 로드
	with open("config.yml","r") as cfgfile:	
		  cfg = yaml.safe_load(cfgfile)

	#shape_GAN 객체 초기화
	# 'device' : GPU or CPU 디바이스 설정. torch.device 객체.
	# 'args' : argparse 전달된 모델 경로 등 실행 파라미터모음
	gan = shape_GAN(cfg, device, args)	#device, args: 참고

	#모델 저장 경로 인자 가져오기
	path_d = args.path_decoder
	path_gid = args.path_gid
	path_gexp = args.path_gexp

	with torch.no_grad(): #그래디언트 계산 비활성화mport yaml				#YAML: 설저파일로딩
from train_gan3d import shape_GAN	#3D 형태 새성 GAN 모델
import os				
from test_gan3d import Test		#GAN 테스트 클래스
import torch				#파이토치 사용하기

from test_texture import test_texture	# 텍스쳐 생성함수
from texture_model.progan_modules import Generator, Discriminator #GAN 기반 텍스쳐 생성기 판별기
from texture_model.assets.texture_processing import add_tex# 
from data.edit_obj import get_mtl
import argparse				#argparse: 머신러닝에서 하이퍼 파라미터 쉽게 넘겨줄 수 있음.



def generate_shapes(num_imgs=1, exp_list=[1,2]):
	"""
 	주어진 개수만큼 얼굴형태 생성후 저장
   	이 함수에서 사용하는 `device` 및 `args` 인자는
    	__main__ 영역에서 argparse로 파싱된 `args`와
    	torch.device로 설정된 `device`를 전역(global)으로 참조합니다.
  	:param num_imgs: 생성 얼굴 샘플개수
   	:param exp_list: 표현 표정 인덱스 리스트
  	"""
	#설정파일 로드
	with open("config.yml","r") as cfgfile:	
		  cfg = yaml.safe_load(cfgfile)

	#shape_GAN 객체 초기화
	# 'device' : GPU or CPU 디바이스 설정. torch.device 객체.
	# 'args' : argparse 전달된 모델 경로 등 실행 파라미터모음
	gan = shape_GAN(cfg, device, args)	#device, args: 참고

	#모델 저장 경로 인자 가져오기
	path_d = args.path_decoder
	path_gid = args.path_gid
	path_gexp = args.path_gexp

	with torch.no_grad(): #그래디언트 계산 비활성화(for inference)

		  test = Test(gan, args)	#Test 클래스 인스턴스 생성
		  test.load_models(path_d=path_d, path_gid=path_gid, path_gexp=path_gexp)	#모델 로드
		  # num_imgs 만큼 반복하며 형태 생성
		  for i in range(num_imgs):
		      #난수 z_id 새로 설정, 각기 다른 아이덴티티 설정
		      test.set_z_id(torch.randn(1,20).to(device))
	              #파읾명으로 i 사용, 옵션 3개에 대한 부울값 설정, exp_list에 표저 지정한거 넣기
		      test.generate(str(i), intensities=True, save_obj=True, render=True, exp_list=exp_list)


def generate_tex(num_imgs=1, exp_list=[1,2]):
	"""
    	주어진 개수만큼 얼굴 텍스처(texture)를 생성하고 저장합니다.
    	:param num_imgs: 생성할 얼굴 샘플 개수
    	:param exp_list: 표현(표정) 인덱스 리스트
    	"""
        # 텍스처 생성을 위한 ProGAN Generator 설정
	input_code_size = 128	#잠재벡터 차원(z_noise)
	channel = 256 # ProGAN내부 채널 크기

	# 2) DataParallel 래퍼로 첫 감싸기
    	#    - 모델 병렬화 준비: .to(device) 전에 호출해야 내부 GPU context가 설정됩니다.
	g_running = Generator(in_channel=channel, input_code_dim=input_code_size+20+20, pixel_norm=False, tanh=False)
	g_running = torch.nn.DataParallel(g_running) #객체 생성 직후 -> DataParallel로 감싸고 결과를 옮김.
	#입력을 여러 GPU에 분할하고 모델의 복사본을만들어 병렬 연산, 출력 집계.
	#간단한 병렬화 용도. 원본 모델에 랩핑.
	
	
	g_running = g_running.to(device)	
	#학습된 체크포인트 가져오기
	model_dir = 'checkpoints/texture_models/'
	number = '142000'
	#generator 파라미터 로드. strict=False로 일부 키 미스 매치 허용)
	g_running.load_state_dict(torch.load(model_dir + 'checkpoint/' + number + '_g.model'), strict=False)
	g_running = torch.nn.DataParallel(g_running) #로드 후의 모델(가중치가 적용된 모델) 랩
	g_running.train(False)#평가 모드로 전환.
	#실제 텍스쳐 생성함수 호출
	test_texture(g_running, num_imgs=num_imgs, exp_list=exp_list, input_code_size=input_code_size, device=device, alpha=1, out_path='results/', zid_dict_path='data/zid_dictionary.pkl')



if __name__ == '__main__':


	### Expressions list for reference

	# ['0_neutral.jpg', '1_smile.jpg', '2_mouth_stretch.jpg', '3_anger.jpg', '4_jaw_left.jpg', '5_jaw_right.jpg', '6_jaw_forward.jpg', '7_mouth_left.jpg', '8_mouth_right.jpg', '9_dimpler.jpg', '10_chin_raiser.jpg', '11_lip_puckerer.jpg', '12_lip_funneler.jpg', '13_sadness.jpg', '14_lip_roll.jpg', '15_grin.jpg', '16_cheek_blowing.jpg', '17_eye_closed.jpg', '18_brow_raiser.jpg', '19_brow_lower.jpg']



	parser = argparse.ArgumentParser()

	parser.add_argument('--results', type=str, default='results/')
	parser.add_argument('--path_decoder', type=str, default='checkpoints/ae/Decoder/2000')
	parser.add_argument('--path_gid', type=str, default='checkpoints/gan3d/Generator_Checkpoint_id/8.0')
	parser.add_argument('--path_gexp', type=str, default='checkpoints/gan3d/Generator_Checkpoint_exp/8.0')
	parser.add_argument('--checkpoints_path', type=str, default='checkpoints/gan3d/')
	parser.add_argument('--zid_dict', type=str, default='data/zid_dictionary.pkl')

	args = parser.parse_args()



	device = torch.device("cuda")
	
	num_imgs = 1
	exp_list = [1,2]    #[Smile, Mouth Stretch]

	print('Generating', num_imgs, 'Faces...')
	
	### GENERATE SHAPES ###
	
	generate_shapes(num_imgs, exp_list)
	
	### GENERATE TEXTURES ###
	
	generate_tex(num_imgs, exp_list)

	#텍스쳐 후처리: MTL 파일 생성.
	add_tex.add_texture_template(in_path='results/', base_path = 'texture_model/assets/texture_processing/base_tex.npy', out_resolution=1024)
		  
	### GENERATE MTL ###	
	
	get_mtl(in_path='results/')  
	
	
		  
    
    
    
