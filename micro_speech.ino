#include <Arduino.h>
#include "PinDefinitionsAndMore.h"
#include <IRremote.hpp>

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <TensorFlowLite.h>

#include "main_functions.h"
#include "audio_provider.h"
#include "command_responder.h"
#include "feature_provider.h"
#include "micro_features_micro_model_settings.h"
#include "micro_features_model.h"
#include "recognize_commands.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
FeatureProvider* feature_provider = nullptr;
RecognizeCommands* recognizer = nullptr;
int32_t previous_time = 0;

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 10 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
int8_t feature_buffer[kFeatureElementCount];
int8_t* model_input_buffer = nullptr;
// IR Sender
const int SEND_PIN = 4;
// LEDS
const uint8_t LED_1_PIN = 12;
const uint8_t LED_2_PIN = 11;
const uint8_t LED_3_PIN = 10;
const uint8_t LED_4_PIN = 9;
// TIME
static unsigned long timer_time_marvin = 0;
static bool in_marvin = false;
static bool in_vol = false;
static bool up = false;

}  // namespace


// The name of this function is important for Arduino compatibility.
void setup() {
  //Before any TF, set up LEDS
  pinMode(LED_1_PIN, OUTPUT);
  pinMode(LED_2_PIN, OUTPUT);
  pinMode(LED_3_PIN, OUTPUT);
  pinMode(LED_4_PIN, OUTPUT);
  digitalWrite(LED_1_PIN, LOW);
  digitalWrite(LED_2_PIN, LOW);
  digitalWrite(LED_3_PIN, LOW);
  digitalWrite(LED_4_PIN, LOW);
  // IR setup
  IrSender.begin();

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver<5> micro_op_resolver(error_reporter); 
  if (micro_op_resolver.AddConv2D() != kTfLiteOk) { 
    return; 
  }
  if (micro_op_resolver.AddDepthwiseConv2D() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddFullyConnected() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddSoftmax() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddReshape() != kTfLiteOk) {
    return;
  }

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  model_input = interpreter->input(0);
  if ((model_input->dims->size != 2) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] !=
       (kFeatureSliceCount * kFeatureSliceSize)) ||
      (model_input->type != kTfLiteInt8)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Bad input tensor parameters in model");
    return;
  }
  model_input_buffer = model_input->data.int8;

  // Prepare to access the audio spectrograms from a microphone or other source
  // that will provide the inputs to the neural network.
  // NOLINTNEXTLINE(runtime-global-variables)
  static FeatureProvider static_feature_provider(kFeatureElementCount,
                                                 feature_buffer);
  feature_provider = &static_feature_provider;

  static RecognizeCommands static_recognizer(error_reporter);
  recognizer = &static_recognizer;

  previous_time = 0;
}

// The name of this function is important for Arduino compatibility.
void loop() {
  // Fetch the spectrogram for the current time.
  const int32_t current_time = LatestAudioTimestamp();
  int how_many_new_slices = 0;
  TfLiteStatus feature_status = feature_provider->PopulateFeatureData(
      error_reporter, previous_time, current_time, &how_many_new_slices);
  if (feature_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Feature generation failed");
    return;
  }
  previous_time = current_time;
  // If no new audio samples have been received since last time, don't bother
  // running the network model.
  if (how_many_new_slices == 0) {
    return;
  }

  // Copy feature buffer to input tensor
  for (int i = 0; i < kFeatureElementCount; i++) {
    model_input_buffer[i] = feature_buffer[i];
  }

  // Run the model on the spectrogram input and make sure it succeeds.
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    return;
  }

  // Obtain a pointer to the output tensor
  TfLiteTensor* output = interpreter->output(0);
  // Determine whether a command was recognized based on the output of inference
  const char* found_command = nullptr;
  uint8_t score = 0;
  bool is_new_command = false;
  TfLiteStatus process_status = recognizer->ProcessLatestResults(
      output, current_time, &found_command, &score, &is_new_command);
  if (process_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "RecognizeCommands::ProcessLatestResults() failed");
    return;
  }
  
  // print_found_command(found_command, "", score);
  if(!in_marvin){
    if(found_command[0] == 'm' && score > 175){
      print_found_command(found_command, "seeking next commands", score);
      // start clock
      timer_time_marvin = millis();
      // turn LEDS on
      all_led_set_fast(HIGH);
      // marvin
      in_marvin = true;
    }
  }
  else { // In Marvin, seek comamands
    unsigned long current_time_marvin = millis();
    if((current_time_marvin - timer_time_marvin) < 6000){ // give usr 6 sec to provide command
      if(!in_vol){
        // OFF
        // ON
        // UP - INIT
        if(found_command == "up" && score > 200){
          print_found_command(found_command, "Seeking next command", score);
          timer_time_marvin = millis(); // extend marvin timer
          in_vol = true;
          up = true;
        }
        // DOWN - INIT
        else if(found_command == "down" && score > 230){
          print_found_command(found_command, "Seeking next command", score);
          timer_time_marvin = millis(); // extend marvin timer
          in_vol = true;
          up = false;
        }
        else if((found_command == "on" && score > 125) || (found_command == "off" && score > 100)){
          print_found_command(found_command, "Sending 0x8", score);
          IrSender.sendNEC(0xF404, 0x8, 0);
          all_led_set_fancy(LOW);
          in_marvin = false;
        }
      }
      // VOLUME CONTROLS - EXEC
      else {
        // up/down one
        if ((found_command == "one" && score > 175)){
          // assume down
          const char* msg = "Sending 0x3";
          int ir_sig = 0x3;
          if(up){
            msg = "Sending 0x2";
            ir_sig = 0x2; 
          }
          print_found_command(found_command, msg, score);
          IrSender.sendNEC(0xF404, ir_sig, 0); // extra delay for first
          delay(400);
          IrSender.sendNEC(0xF404, ir_sig, 0);
          all_led_set_fancy(LOW);
          in_marvin = false;
          in_vol = false;
        }
        // up/down five
        if ((found_command == "five" && score > 145)){
          const char* msg = "Sending 0x3";
          int ir_sig = 0x3;
          if(up){
            msg = "Sending 0x2";
            ir_sig = 0x2; 
          }
          print_found_command(found_command, msg, score);
          IrSender.sendNEC(0xF404, ir_sig, 0); // for my monitor, requires longer delay for first click? thanks LG
          delay(400);
          for(int i = 0; i < 4; i++){
            IrSender.sendNEC(0xF404, ir_sig, 0);
            delay(250); // min between ir sends is 5
          }
          IrSender.sendNEC(0xF404, ir_sig, 0); // ensure no delay on last
          all_led_set_fancy(LOW);
          in_marvin = false;
          in_vol = false;
        }
      }
    } else{
      // if usr runs out of time to respond, display shutoff.
        in_marvin = false;
        for(int i = 0; i < 5; i++){
          all_led_set_fast(LOW);
          delay(200);
          all_led_set_fast(HIGH);
          delay(200);
        }
        all_led_set_fancy(LOW);
      }
  }
}

void print_found_command(const char* command, const char* msg, const int score){
  Serial.print("Heard ");
  Serial.print(command);
  Serial.print(" - ");
  Serial.print(msg);
  Serial.print(" (");
  Serial.print(score);
  Serial.println(")");
}
void all_led_set_fast(int state){
  digitalWrite(LED_1_PIN, state);
  digitalWrite(LED_2_PIN, state);
  digitalWrite(LED_3_PIN, state);
  digitalWrite(LED_4_PIN, state);
}

void all_led_set_fancy(int state){
  digitalWrite(LED_1_PIN, state);
  delay(150);
  digitalWrite(LED_2_PIN, state);
  delay(150);
  digitalWrite(LED_3_PIN, state);
  delay(150);
  digitalWrite(LED_4_PIN, state);
}