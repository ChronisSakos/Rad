"""Example usage:

python3 examples/matmul_tpu.py \
  --model test_data/model_matmul_64_quant_edgetpu.tflite  \
  --c 10
"""

import argparse
import time
import numpy as np
import sys
import give_inputs_matmul
from PIL import Image
from PIL import ImageDraw
from csv import writer
from numpy import genfromtxt

from PIL import Image
from pycoral.adapters import classify, detect
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter


def draw_objects(draw, objs, labels):
  """Draws the bounding box and label for each object."""
  for obj in objs:
    bbox = obj.bbox
    draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                   outline='red')
    draw.text((bbox.xmin + 10, bbox.ymin + 10),
              '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
              fill='red')


def set_input_tensor(interpreter, *args):
  input_details = interpreter.get_input_details()
  tensor_index = [None]*2
  input_tensor = [None]*2

  for i in range(len(args)): 
    tensor_index[i] = input_details[i]['index']
    input_tensor[i] = interpreter.tensor(tensor_index[i])()[0]
    input_tensor[i][:] = args[i]
    #print("TPU Input",i+1,":", input_tensor[i])


def classify_image(interpreter):
  output_details = [None]*1
  output = [None]*1

  for i in range(1):
    output_details[i] = interpreter.get_output_details()[i]
    output[i] = interpreter.get_tensor(output_details[i]['index'])
    scale, zero_point = output_details[i]['quantization']
    scale = np.float32(255*64)    
    
  start = time.perf_counter() 
  interpreter.invoke() 
  inference_time = time.perf_counter() - start
  return output, scale, inference_time


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-m1', '--model1', required=True,
                      help='File path of .tflite file.')
  parser.add_argument('-m2', '--model2', required=True,
                      help='File path of .tflite file.')
  parser.add_argument('-m3', '--model3', required=True,
                      help='File path of .tflite file.')
  parser.add_argument('-m4', '--model4', required=True,
                      help='File path of .tflite file.') 
  parser.add_argument('-m5', '--model5', required=True,
                      help='File path of .tflite file.')
  parser.add_argument('-m6', '--model6', required=True,
                      help='File path of .tflite file.') 
  parser.add_argument('-l1', '--labels1',
                      help='File path of labels file.')
  parser.add_argument('-l2', '--labels2',
                      help='File path of labels file.')                                                                  
  parser.add_argument('-i1', '--input1', required=True,
                      help='Image to be classified.')
  parser.add_argument('-i2', '--input2', required=True,
                      help='Image to be classified.')
  parser.add_argument('-o', '--output', required=True,
                      help='Image to be classified.')                          
  parser.add_argument('-k', '--top_k', type=int, default=1,
                      help='Max number of classification results')
  parser.add_argument('-t1', '--threshold1', type=float, default=0.0,
                      help='Classification score threshold')
  parser.add_argument('-t2', '--threshold2', type=float, default=0.4,
                      help='Classification score threshold')  
  parser.add_argument('-c', '--count', type=int, default=5,
                      help='Number of times to run inference')
  parser.add_argument('-n1', '--num1', type=int, default=1,
                      help='Number of executions')
  parser.add_argument('-n2', '--num2', type=int, default=1,
                      help='Number of executions 2')
  parser.add_argument('-n3', '--num3', type=int, default=1,
                      help='Number of executions')
  parser.add_argument('-n4', '--num4', type=int, default=1,
                      help='Number of executions 2')
  parser.add_argument('-n5', '--num5', type=int, default=1,
                      help='Number of executions')
  parser.add_argument('-n6', '--num6', type=int, default=1,
                      help='Number of executions 2')                        
  args = parser.parse_args()
  
  
  """Load matrix A & B"""
  my_data = genfromtxt('data.csv')
  
  x1 = my_data[0:4096]
  x2 = my_data[4096:]
  
  size=(1,1,64,64)
  
  matrix_A = np.reshape(x1,size) 
  matrix_B = np.reshape(x2,size)  

  """While Loop"""
  
  N1 = int(args.num1)
  N2 = int(args.num2)
  N3 = int(args.num3)
  M1 = int(args.num4)
  M2 = int(args.num5)
  M3 = int(args.num6)
 
  reference = []
  with open('matmul_golden_reference_output.txt') as f_ref:
      for line in f_ref:
          reference.append(line)
  sampling_count = 0

  while(1):
      aa = input()
      if str(aa)=='d':
          sampling_count += 1
          print('\nSampling Count', sampling_count,'\n')
          print('========== T P U ==========\n')

          """---------------- MATMUL ----------------"""        
          matmul_infer = []
          count_matmul_array = []
          for i in range(N1):
              interpreter = make_interpreter(*args.model1.split('@'))
              interpreter.allocate_tensors()

              tpu = args.model1.split('.')[0].split('_')[-1] 

              set_input_tensor(interpreter, matrix_A, matrix_B)

              for i in range(args.count): 
                  pred, scale, inference_time = classify_image(interpreter)

                  if (i-1)==0:
                      infer_time = np.round((inference_time*1000),2)
                      #print('Inference Time: %.2f ms' % (inference_time * 1000))

                      if str(tpu)=='quant':
                          with open('matmul_d_time_ARM.txt',"a") as f1:
                              f1.write(str(infer_time)+"\n")
                      elif str(tpu)=='edgetpu':
                          with open('matmul_d_time_TPU.txt',"a") as f2:
                              f2.write(str(infer_time)+"\n")

              matmul_infer.append(infer_time)   

              out = [None]*1

              out = np.uint64(np.round(pred,0))*scale
              out1 = np.reshape(out,(64*64))
              
              count_matmul = 0

              for i in range(64*64):
                if float(out1[i]) == float(reference[i]):
                  count_matmul += 1
                  
              count_matmul_array.append(count_matmul)
        
              #print(tpu,"Output",args.num1,'->',out) 
  
              for i in range(64*64):
                  if str(tpu)=='quant':
                      with open('matmul_d_results_ARM.txt',"a") as f3:
                          f3.write(str(out1[i])+"\n")
                  elif str(tpu)=='edgetpu':
                      with open('matmul_d_results_TPU.txt',"a") as f4:
                          f4.write(str(out1[i])+"\n")

          print('--------MatMul--------')
          for k in range(N1):
                print('Iteration',k+1,'-> %.2f ms,' % matmul_infer[k],'Error_rate: %.2f' % ((4096-count_matmul_array[k])*100/4096),"%") 
          print('Min Time: %.2f ms' % min(*matmul_infer))
          print('Average Time: %.2f ms' % (sum(matmul_infer)/N1))
          print('Max Time: %.2f ms' %max(*matmul_infer))

          """-------------- SHIP DETECTION -----------------"""        
          count_ship = 0
          ship_infer = []

          for i in range(N2):
              labels = read_label_file(args.labels1) if args.labels1 else {} 

              interpreter = make_interpreter(*args.model2.split('@'))
              interpreter.allocate_tensors()

              size = common.input_size(interpreter)
              image = Image.open(args.input1).convert('RGB').resize(size, Image.ANTIALIAS)
              common.set_input(interpreter, image) 
                
              tpu = args.model2.split('.')[0].split('_')[-1] 

              for i in range(args.count):
                start = time.perf_counter()
                interpreter.invoke()
                inference_time = time.perf_counter() - start
                classes = classify.get_classes(interpreter, args.top_k, args.threshold1)

                if (i-1)==0:
                  infer_time = np.round((inference_time*1000),2)
                  #print('Inference time: %.2f ms' % (inference_time * 1000))

                  if str(tpu)=='quant':
                    with open('ship_time_ARM.txt',"a") as f1:
                      f1.write(str(infer_time)+"\n")
                  elif str(tpu)=='edgetpu':
                    with open('ship_time_TPU.txt',"a") as f2:
                      f2.write(str(infer_time)+"\n")

              ship_infer.append(infer_time)           

              for c in classes:
                pred = np.round(c.score,3)
                #print('Prediction',N2,'-> %.3f' % (c.score))
                if pred == 0.574:
                  count_ship += 1

                if str(tpu)=='quant':
                  with open('ship_results_ARM.txt',"a") as f3:
                    f3.write(str(pred)+"\n")
                elif str(tpu)=='edgetpu':
                  with open('ship_results_TPU.txt',"a") as f4:
                    f4.write(str(pred)+"\n")

          print('--------SHIP--------')
          print('Min Time: %.2f ms' % min(*ship_infer))
          print('Average Time: %.2f ms' % (sum(ship_infer)/N2))
          print('Max Time: %.2f ms' %max(*ship_infer))
          print('Correct results:', count_ship, 'out of', N2)


          """-------------- OBJECT DETECTION -----------------"""        
          count_object_1 = 0
          count_object_2 = 0
          count_object_3 = 0
          object_infer = []          

          for i in range(N3):
              labels = read_label_file(args.labels2) if args.labels2 else {}
              interpreter = make_interpreter(args.model3)
              interpreter.allocate_tensors()

              image = Image.open(args.input2)
              _, scale = common.set_resized_input(
                interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))


              tpu = args.model3.split('.')[0].split('_')[-1]
      

              for i in range(args.count):
                start = time.perf_counter()
                interpreter.invoke()
                inference_time = time.perf_counter() - start
                objs = detect.get_objects(interpreter, args.threshold2, scale)

                if (i-1)==0:
                  infer_time = np.round((inference_time*1000),2)
                  #print('Inference time: %.2f ms' % (inference_time*1000))

                  if str(tpu)=='ptq':
                    with open('object_time_ARM.txt',"a") as f1:
                      f1.write(str(infer_time)+"\n")
                  elif str(tpu)=='edgetpu':
                    with open('object_time_TPU.txt',"a") as f2:
                      f2.write(str(infer_time)+"\n")
     
              object_infer.append(infer_time) 

              if not objs:
                print('No objects detected')

              for obj in objs:
                label = labels.get(obj.id, obj.id)
                score = np.round(obj.score,3)
                box = obj.bbox
                #print('Prediction',N3,'->',label, score, box)

              if (str(label) == str('bird')):
                count_object_1 += 1 
              if (score == 0.871):
                count_object_2 += 1
              if ((obj.bbox.xmin == 85) and (obj.bbox.xmax == 471) and (obj.bbox.ymin == 109) and (obj.bbox.ymax == 914)):  
                count_object_3 += 1

              if str(tpu)=='ptq':
                with open('object_results_ARM.txt',"a") as f3:
                  f3.write(str(label)+' '+str(score)+' '+str(box)+"\n")
              elif str(tpu)=='edgetpu':
                with open('object_results_TPU.txt',"a") as f4:
                  f4.write(str(label)+' '+str(score)+' '+str(box)+"\n")
        
              if args.output:
                image = image.convert('RGB')
                draw_objects(ImageDraw.Draw(image), objs, labels)
                image.save(args.output)
                image.show()

          print('-----Object Detection-----')
          print('Min Time: %.2f ms' % min(*object_infer))
          print('Average Time: %.2f ms' % (sum(object_infer)/N3))
          print('Max Time: %.2f ms' %max(*object_infer))
          print('Correct labels:', count_object_1, 'out of', N3)
          print('Correct prediction:', count_object_2, 'out of', N3)          
          print('Correct output images:', count_object_3, 'out of', N3)


          print('\n========== A R M ==========\n')
          """---------------- ARM ---------------"""
          """---------------- MATMUL ----------------"""        
          matmul_infer = []
          count_matmul_array = []
          for i in range(M1):
              interpreter = make_interpreter(*args.model4.split('@'))
              interpreter.allocate_tensors()

              tpu = args.model4.split('.')[0].split('_')[-1] 

              set_input_tensor(interpreter, matrix_A, matrix_B)

              for i in range(args.count): 
                  pred, scale, inference_time = classify_image(interpreter)

                  if (i-1)==0:
                      infer_time = np.round((inference_time*1000),2)
                      #print('Inference Time: %.2f ms' % (inference_time * 1000))

                      if str(tpu)=='quant':
                          with open('matmul_d_time_ARM.txt',"a") as f1:
                              f1.write(str(infer_time)+"\n")
                      elif str(tpu)=='edgetpu':
                          with open('matmul_d_time_TPU.txt',"a") as f2:
                              f2.write(str(infer_time)+"\n")

              matmul_infer.append(infer_time)                  

              out = [None]*1

              out = np.uint64(np.round(pred,0))*scale
              out1 = np.reshape(out,(64*64))

              count_matmul = 0

              for i in range(64*64):
                if float(out1[i]) == float(reference[i]):
                  count_matmul += 1
              count_matmul_array.append(count_matmul)

              #print(tpu,"Output",args.num1,'->',out) 
  
              for i in range(64*64):
                  if str(tpu)=='quant':
                      with open('matmul_d_results_ARM.txt',"a") as f3:
                          f3.write(str(out1[i])+"\n")
                  elif str(tpu)=='edgetpu':
                      with open('matmul_d_results_TPU.txt',"a") as f4:
                          f4.write(str(out1[i])+"\n")

          print('--------MatMul--------')
          for k in range(M1):
                print('Iteration',k+1,'-> %.2f ms,' % matmul_infer[k],'Error_rate: %.2f' % ((4096-count_matmul_array[k])*100/4096),"%") 
          print('Min Time: %.2f ms' % min(*matmul_infer))
          print('Average Time: %.2f ms' % (sum(matmul_infer)/M1))
          print('Max Time: %.2f ms' %max(*matmul_infer))

          """-------------- SHIP DETECTION -----------------"""        
          count_ship = 0
          ship_infer = []
          for i in range(M2):
              labels = read_label_file(args.labels1) if args.labels1 else {} 

              interpreter = make_interpreter(*args.model5.split('@'))
              interpreter.allocate_tensors()

              size = common.input_size(interpreter)
              image = Image.open(args.input1).convert('RGB').resize(size, Image.ANTIALIAS)
              common.set_input(interpreter, image) 
                
              tpu = args.model5.split('.')[0].split('_')[-1] 

              for i in range(args.count):
                start = time.perf_counter()
                interpreter.invoke()
                inference_time = time.perf_counter() - start
                classes = classify.get_classes(interpreter, args.top_k, args.threshold1)

                if (i-1)==0:
                  infer_time = np.round((inference_time*1000),2)
                  #print('Inference time: %.2f ms' % (inference_time * 1000))

                  if str(tpu)=='quant':
                    with open('ship_time_ARM.txt',"a") as f1:
                      f1.write(str(infer_time)+"\n")
                  elif str(tpu)=='edgetpu':
                    with open('ship_time_TPU.txt',"a") as f2:
                      f2.write(str(infer_time)+"\n")

              ship_infer.append(infer_time)           

              for c in classes:
                pred = np.round(c.score,3)
                #print('Prediction',M2,'-> %.3f' % (c.score))
                if pred == 0.574:
                  count_ship += 1

                if str(tpu)=='quant':
                  with open('ship_results_ARM.txt',"a") as f3:
                    f3.write(str(pred)+"\n")
                elif str(tpu)=='edgetpu':
                  with open('ship_results_TPU.txt',"a") as f4:
                    f4.write(str(pred)+"\n")

          print('--------SHIP--------')
          print('Min Time: %.2f ms' % min(*ship_infer))
          print('Average Time: %.2f ms' % (sum(ship_infer)/M2))
          print('Max Time: %.2f ms' %max(*ship_infer))
          print('Correct results:', count_ship, 'out of', M2)

          """-------------- OBJECT DETECTION -----------------"""        
          count_object_1 = 0
          count_object_2 = 0
          count_object_3 = 0
          object_infer = []          

          for i in range(M3):
              labels = read_label_file(args.labels2) if args.labels2 else {}
              interpreter = make_interpreter(args.model6)
              interpreter.allocate_tensors()

              image = Image.open(args.input2)
              _, scale = common.set_resized_input(
                interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))


              tpu = args.model6.split('.')[0].split('_')[-1]
      

              for i in range(args.count):
                start = time.perf_counter()
                interpreter.invoke()
                inference_time = time.perf_counter() - start
                objs = detect.get_objects(interpreter, args.threshold2, scale)

                if (i-1)==0:
                  infer_time = np.round((inference_time*1000),2)
                  #print('Inference time: %.2f ms' % (inference_time*1000))

                  if str(tpu)=='ptq':
                    with open('object_time_ARM.txt',"a") as f1:
                      f1.write(str(infer_time)+"\n")
                  elif str(tpu)=='edgetpu':
                    with open('object_time_TPU.txt',"a") as f2:
                      f2.write(str(infer_time)+"\n")
     
              object_infer.append(infer_time) 

              if not objs:
                print('No objects detected')

              for obj in objs:
                label = labels.get(obj.id, obj.id)
                score = np.round(obj.score,3)
                box = obj.bbox
                #print('Prediction',M3,'->',label, score, box)

              if (str(label) == str('bird')):
                count_object_1 += 1 
              if (score == 0.879):
                count_object_2 += 1
              if ((obj.bbox.xmin == 89) and (obj.bbox.xmax == 468) and (obj.bbox.ymin == 109) and (obj.bbox.ymax == 914)):  
                count_object_3 += 1

              if str(tpu)=='ptq':
                with open('object_results_ARM.txt',"a") as f3:
                  f3.write(str(label)+' '+str(score)+' '+str(box)+"\n")
              elif str(tpu)=='edgetpu':
                with open('object_results_TPU.txt',"a") as f4:
                  f4.write(str(label)+' '+str(score)+' '+str(box)+"\n")
        
              if args.output:
                image = image.convert('RGB')
                draw_objects(ImageDraw.Draw(image), objs, labels)
                image.save(args.output)
                image.show()

          print('-----Object Detection-----')
          print('Min Time: %.2f ms' % min(*object_infer))
          print('Average Time: %.2f ms' % (sum(object_infer)/M3))
          print('Max Time: %.2f ms' %max(*object_infer))
          print('Correct labels:', count_object_1, 'out of', M3)
          print('Correct prediction:', count_object_2, 'out of', M3)          
          print('Correct output images:', count_object_3, 'out of', M3)

          print('\nSampling',sampling_count,'Finished! Press d for the next one...\n')
	
      elif str(aa)=='q':
          break

if __name__ == '__main__':
  main()
