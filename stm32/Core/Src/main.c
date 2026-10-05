/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI // not in C standard; define if the toolchain omits it
#define M_PI 3.14159265359f
#endif
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
// parameters
#define SAMPLING_RATE 100 // Hz, sensor sampling frequency
#define LPF_CUTOFF 10 // Hz, lowpass cutoff
#define LPF_ORDER 64 // FIR filter order
#define NUMTAPS (LPF_ORDER + 1) // FIR taps
#define PE_ORDER 5 // embedding dimension
#define PE_NHASH 3125 // pattern buckets = PE_ORDER^PE_ORDER; UPDATE IF PE_ORDER CHANGES
#define PE_NPERM 120 // distinct patterns = PE_ORDER! (entropy normalizer); UPDATE IF PE_ORDER CHANGES
#define PE_DELAY (SAMPLING_RATE / (2 * LPF_CUTOFF)) // delay (Nyquist-matched)
#define PE_W 1000 // entropy window size in samples
#define PE_S 100 // window step size in samples
#define K 10 // shrinkage strength: pseudo-windows pulling the running mean toward the sober baseline
#define PE_EMBED (PE_W - (PE_ORDER - 1) * PE_DELAY) // PE embedded vectors
#define N_FEATURES 7 // ax, ay, az, rx, ry, rz, throttle
#define N_FILTERED 6 // accel(0..2) + gyro(3..5) get FIR; throttle(6) does not
#define N_CYCLES 1000 // timed cycles (after window-filling warm-up)
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

COM_InitTypeDef BspCOMInit;

/* USER CODE BEGIN PV */
// per-channel raw sensor range [lo, hi]
static const float feat_lo[N_FEATURES] = {-160, -160, -160, -2000, -2000, -2000, 0};
static const float feat_hi[N_FEATURES] = {160, 160, 160, 2000, 2000, 2000, 255};

// model state (synthetic, fixed)
static float fir_taps[NUMTAPS];
static float fir_hist[N_FILTERED][NUMTAPS]; // per-channel FIR delay line
static float win[N_FEATURES][PE_W]; // sliding filtered window
static float wsc_mu[N_FEATURES]; // within-subject centering means
static float sc_mean[N_FEATURES]; // StandardScaler mean
static float sc_scale[N_FEATURES]; // StandardScaler scale
static float lr_coef[N_FEATURES]; // LogisticRegression coefficients
static float lr_b; // LogisticRegression intercept
static float m0; // sober baseline: shrinkage target for the running mean
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_ICACHE_Init(void);
/* USER CODE BEGIN PFP */
static void init_synthetic_state(void);
static void gen_samples(float raw_block[N_FEATURES][PE_S]);
static float run_cycle(float raw_block[N_FEATURES][PE_S], float *sum_lr, int *count);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_ICACHE_Init();
  /* USER CODE BEGIN 2 */

  /* USER CODE END 2 */

  /* Initialize leds */
  BSP_LED_Init(LED_GREEN);

  /* Initialize USER push-button, will be used to trigger an interrupt each time it's pressed.*/
  BSP_PB_Init(BUTTON_USER, BUTTON_MODE_EXTI);

  /* Initialize COM1 port (115200, 8 bits (7-bit data + 1 stop bit), no parity */
  BspCOMInit.BaudRate   = 115200;
  BspCOMInit.WordLength = COM_WORDLENGTH_8B;
  BspCOMInit.StopBits   = COM_STOPBITS_1;
  BspCOMInit.Parity     = COM_PARITY_NONE;
  BspCOMInit.HwFlowCtl  = COM_HWCONTROL_NONE;
  if (BSP_COM_Init(COM1, &BspCOMInit) != BSP_ERROR_NONE)
  {
    Error_Handler();
  }

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  setvbuf(stdout, NULL, _IONBF, 0); // unbuffered so printf appears immediately

  // init
  init_synthetic_state();
  float sum_lr = 0.0f; // running sum of P(impaired) for the ride mean
  int count = 0; // windows seen so far
  static float raw_block[N_FEATURES][PE_S]; // newest pe_s raw samples per channel

  // warm-up to fill the entropy window (untimed)
  for (int i = 0; i < PE_W / PE_S; i++) {
    gen_samples(raw_block);
    run_cycle(raw_block, &sum_lr, &count);
  }

  // timed cycles: accumulate running stats (sum, sum of squares, min, max)
  double sum = 0.0, sum_sq = 0.0, mn = DBL_MAX, mx = 0.0; // mn starts at max so any cycle is smaller
  volatile float sink = 0.0f; // keeps run_cycle's output live (see return comment)

  for (int i = 0; i < N_CYCLES; i++) { // for test cycles ...
    gen_samples(raw_block); // generate sensor data outside the timed region
    uint32_t t0 = HAL_GetTick(); // start the per-cycle timer (SysTick, ms)
    sink = run_cycle(raw_block, &sum_lr, &count); // run a cycle
    double ms = (uint32_t)(HAL_GetTick() - t0); // elapsed cycle time in ms
    sum += ms; sum_sq += ms * ms; // feed mean/std accumulators
    if (ms < mn) mn = ms; // track fastest cycle
    if (ms > mx) mx = ms; // track slowest cycle
    printf("cycle %d: %.0f ms\r\n", i, ms); // per-cycle progress (outside the timed region)
  }
  (void)sink; // read once so the volatile write isn't flagged as unused

  // reduce running stats to mean and std (std = sqrt(E[x^2] - E[x]^2))
  double mean = sum / N_CYCLES;
  double sd = sqrt(sum_sq / N_CYCLES - mean * mean);

  // report the run config and the per-cycle latency distribution
  printf("cycles=%d  features=%d  pe_w=%d  pe_s=%d\r\n",
         N_CYCLES, N_FEATURES, PE_W, PE_S);
  printf("per-cycle latency: mean=%.2f ms  std=%.2f ms  min=%.0f ms  max=%.0f ms\r\n",
         mean, sd, mn, mx);

  while (1)
  {

    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);

  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSIDiv = RCC_HSI_DIV2;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_NONE;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2
                              |RCC_CLOCKTYPE_PCLK3;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_HSI;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure the programming delay
  */
  __HAL_FLASH_SET_PROGRAM_DELAY(FLASH_PROGRAMMING_DELAY_0);
}

/**
  * @brief ICACHE Initialization Function
  * @param None
  * @retval None
  */
static void MX_ICACHE_Init(void)
{

  /* USER CODE BEGIN ICACHE_Init 0 */

  /* USER CODE END ICACHE_Init 0 */

  /* USER CODE BEGIN ICACHE_Init 1 */

  /* USER CODE END ICACHE_Init 1 */

  /** Enable instruction cache (default 2-ways set associative cache)
  */
  if (HAL_ICACHE_Enable() != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN ICACHE_Init 2 */

  /* USER CODE END ICACHE_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin : PC4 */
  GPIO_InitStruct.Pin = GPIO_PIN_4;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */
// uniform in [0, 1); generator value is irrelevant to timing
static float urand(void) {
    return rand() / (RAND_MAX + 1.0f);
}

// scale a [0, 1) draw into an arbitrary [lo, hi) range
static float urange(float lo, float hi) { return lo + (hi - lo) * urand(); }

// fill all model state with synthetic values (shape/size matters, exact values don't)
static void init_synthetic_state(void) {
    // init normalized FIR taps (hamming-shaped)
    float s = 0.0f;
    for (int k = 0; k < NUMTAPS; k++) {
        fir_taps[k] = 0.54f - 0.46f * cosf(2.0f * M_PI * k / (NUMTAPS - 1));
        s += fir_taps[k];
    }
    for (int k = 0; k < NUMTAPS; k++) {
        fir_taps[k] /= s;
    }

    // init WSC means
    for (int j = 0; j < N_FEATURES; j++) {
        wsc_mu[j] = urange(-0.1, 0.1);
    }

    // init scaler and LR
    for (int j = 0; j < N_FEATURES; j++) {
        sc_mean[j]  = urange(0.3, 0.7);  // StandardScaler mean
        sc_scale[j] = urange(0.05, 0.2); // StandardScaler scale (nonzero)
        lr_coef[j]  = urange(-2.0, 2.0); // LR coefficients
    }
    lr_b = urange(-1.0, 1.0); // LR intercept
    m0 = urange(0.0, 1.0); // sober baseline P(impaired) shrinkage target
}

// normalized permutation entropy
static float perm_entropy(const float *x) {
    // init pattern histogram
    static int counts[PE_NHASH];
    memset(counts, 0, sizeof(counts));

    // build each embedding -> ordinal pattern -> histogram
    for (int i = 0; i < PE_EMBED; i++) {
        float v[PE_ORDER]; // embedded vector: x[i], x[i+delay], ...
        int idx[PE_ORDER];  // identity start; argsorted below into the ordinal pattern

        // gather the delay-embedded samples and seed the index array
        for (int k = 0; k < PE_ORDER; k++) {
            v[k] = x[i + k * PE_DELAY];
            idx[k] = k;
        }

        // argsort v -> idx holds the ordinal pattern (insertion sort; order is tiny)
        for (int a = 1; a < PE_ORDER; a++) {
            int key = idx[a];
            float kv = v[key];
            int b = a - 1;
            while (b >= 0 && v[idx[b]] > kv) {
                idx[b + 1] = idx[b];
                b--;
            }
            idx[b + 1] = key;
        }

        // encode the permutation as a base-order integer (unique per pattern)
        int h = 0;
        for (int k = 0; k < PE_ORDER; k++) {
            h = h * PE_ORDER + idx[k];
        }

        counts[h]++; // increment this pattern's count
    }

    // Shannon entropy of the pattern distribution
    float pe = 0.0f;
    for (int h = 0; h < PE_NHASH; h++) {
        if (counts[h]) { // skip empty buckets (0 * log0 = 0, and log0 is undefined)
            float p = (float)counts[h] / PE_EMBED; // pattern's relative frequency
            pe -= p * log2f(p); // accumulate -sum p*log2(p)
        }
    }
    return pe / log2f((float)PE_NPERM); // normalize by log2(order!) -> [0, 1]
}

// sigmoid function
static float sigmoid(float z) { return 1.0f / (1.0f + expf(-z)); }

// generate the next pe_s synthetic sensor samples (kept outside the timed cycle)
static void gen_samples(float raw_block[N_FEATURES][PE_S]) {
    for (int c = 0; c < N_FEATURES; c++) {
        for (int n = 0; n < PE_S; n++) {
            raw_block[c][n] = urange(feat_lo[c], feat_hi[c]);
        }
    }
}

// causal FIR low-pass: push one sample into the channel's delay line, return output
static float fir_step(float *hist, float raw) {
    for (int k = NUMTAPS - 1; k > 0; k--) {
        hist[k] = hist[k - 1];
    }
    hist[0] = raw;
    float out = 0.0f;
    for (int k = 0; k < NUMTAPS; k++) {
        out += fir_taps[k] * hist[k];
    }
    return out;
}

// runs one detection cycle
static float run_cycle(float raw_block[N_FEATURES][PE_S], float *sum_lr, int *count) {
    // slide each window left by pe_s, filter pe_s new samples into the tail
    for (int c = 0; c < N_FEATURES; c++) { // for each feature ...
        // consecutive windows overlap by PE_W - PE_S samples; drop the oldest
        // PE_S by shifting the kept samples to the front, freeing the tail
        memmove(win[c], win[c] + PE_S, (PE_W - PE_S) * sizeof(float));
        // fill the freed tail (last PE_S slots) with the new filtered samples
        for (int n = 0; n < PE_S; n++) {
            // accel/gyro get the FIR low-pass; throttle passes through
            float out = raw_block[c][n];
            if (c < N_FILTERED) {
                out = fir_step(fir_hist[c], out); // run the FIR filter (history persists across cycles)
            }
            win[c][PE_W - PE_S + n] = out;
        }
    }

    // per-channel permutation entropy -> feature vector, then WSC centering
    float x[N_FEATURES];
    for (int c = 0; c < N_FEATURES; c++) {
        x[c] = perm_entropy(win[c]) - wsc_mu[c];
    }

    // per-window score: StandardScaler + LogisticRegression -> P(impaired|x)
    float z = lr_b; // start from the LR intercept (bias term)
    for (int c = 0; c < N_FEATURES; c++) {
        // standardize each feature, then accumulate its weighted contribution
        z += lr_coef[c] * (x[c] - sc_mean[c]) / sc_scale[c];
    }
    float s_t = sigmoid(z); // convert the logit to a probability in (0, 1)

    // shrinkage running-mean update: K pseudo-windows pull early estimates toward
    // the sober baseline m0, damping small-sample spikes. The alarm (mean >
    // conformal threshold) uses a threshold calibrated offline, so only that
    // single comparison runs on-device.
    *sum_lr += s_t;
    float mean_lr = (*sum_lr + K * m0) / (float)(++(*count) + K);

    return mean_lr; // returned only to feed the volatile sink (prevents dead-code elimination)
}
/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
