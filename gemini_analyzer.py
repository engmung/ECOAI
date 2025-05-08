import os
import asyncio
import logging
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# 요청 제한 관리를 위한 세마포어
# 1분당 최대 2개의 요청 허용
API_SEMAPHORE = asyncio.Semaphore(2)
API_RATE_LIMIT_SECONDS = 60  # 1분 딜레이

async def analyze_script_with_gemini(script: str, video_title: str, channel_name: str) -> str:
    """
    Gemini API를 사용하여 스크립트를 분석하고 마크다운 보고서를 생성합니다.
    
    Args:
        script: 분석할 유튜브 스크립트
        video_title: 영상 제목
        channel_name: 채널명
    
    Returns:
        마크다운 형식의 분석 보고서 또는 오류 시 오류 메시지
    """
    try:
        # API 키 설정
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
            return "# AI 분석 보고서\n\n## 분석 오류\n\nGEMINI_API_KEY 환경 변수가 설정되지 않았습니다."
        
        # 프롬프트 작성 - 간소화된 프롬프트
        prompt = f"""# 경제 영상 스크립트 분석 요청

제목: {video_title}
채널: {channel_name}

이 영상의 스크립트를 분석해주세요."""

        # 비동기적으로 Gemini API 호출 (API 제한 고려)
        async with API_SEMAPHORE:
            logger.info(f"Gemini API 호출 시작: {video_title}")
            
            # API 호출 전 타임스탬프 기록
            start_time = asyncio.get_event_loop().time()
            
            # 프로세스 시작
            def call_gemini():
                try:
                    client = genai.Client(api_key=api_key)
                    model = "gemini-2.5-flash-preview-04-17"  # 최신 모델 사용
                    
                    # Content 객체 생성
                    contents = [
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=prompt)],
                        ),
                    ]
                    
                    # 시스템 지시사항 설정 - 마크다운 형식 강조
                    system_instruction = """너는 뛰어난 경제학자로서 제공된 영상 스크립트를 분석해 보고서를 작성해주는 역할을 맡았습니다.  
                    
다음 지침을 반드시 따르세요:

1. 보고서는 "# AI 분석 보고서"로 시작하세요.

2. 각 섹션 제목은 ## 형식(두 개의 해시태그)으로 작성합니다:
   - ## 요약
   - ## 주요 키워드
   - ## 핵심 내용 분석
   - ## 경제적 시사점
   - ## 추가 참고사항

3. 불릿 포인트는 일관되게 다음과 같이 작성하세요:
   - 첫 번째 항목
   - 두 번째 항목

4. 강조하고 싶은 부분은 **두 개의 별표**로 감싸세요.

5. 문단과 문단 사이는 빈 줄로 구분하세요.

6. 각 하위 섹션은 번호를 매겨 구분하세요:
   1. 첫 번째 하위 섹션
   2. 두 번째 하위 섹션

7. 중요한 구분점에는 --- 를 사용하세요.

8. 정확한 수치 데이터가 없을 경우 추측하지 말고 "정확한 수치가 제공되지 않음"으로 표기하세요."""
                    
                    generate_content_config = types.GenerateContentConfig(
                        response_mime_type="text/plain",
                        system_instruction=[types.Part.from_text(text=system_instruction)],
                    )
                    
                    # 스트리밍 응답 수집
                    response_text = ""
                    
                    # 스크립트를 분석에 사용
                    full_prompt = f"{prompt}\n\n스크립트 내용:\n{script}"
                    contents[0].parts[0].text = full_prompt
                    
                    for chunk in client.models.generate_content_stream(
                        model=model,
                        contents=contents,
                        config=generate_content_config,
                    ):
                        if chunk.text:
                            response_text += chunk.text
                    
                    return response_text
                except Exception as e:
                    logger.error(f"Gemini 함수 내 오류: {str(e)}")
                    return f"# AI 분석 보고서\n\n## Gemini API 오류\n\n{str(e)}"
            
            # 비동기적으로 API 호출 실행
            try:
                response_text = await asyncio.to_thread(call_gemini)
            except Exception as e:
                logger.error(f"asyncio.to_thread 오류: {str(e)}")
                return f"# AI 분석 보고서\n\n## 분석 오류\n\nasyncio.to_thread 실행 중 오류가 발생했습니다: {str(e)}"
            
            # API 호출 후 경과 시간 계산
            elapsed_time = asyncio.get_event_loop().time() - start_time
            # 1분에서 경과 시간을 뺀 만큼 대기 (최소 0초)
            wait_time = max(0, API_RATE_LIMIT_SECONDS - elapsed_time)
            
            if wait_time > 0:
                logger.info(f"API 제한 준수를 위해 {wait_time:.1f}초 대기")
                await asyncio.sleep(wait_time)
            
            if response_text:
                logger.info("Gemini 분석 완료")
                
                # 응답이 마크다운 형식인지 확인하고 수정
                if not response_text.startswith("# AI 분석 보고서"):
                    response_text = "# AI 분석 보고서\n\n" + response_text
                
                # 마크다운 형식 일관성 개선
                response_text = clean_markdown_format(response_text)
                
                return response_text
            else:
                logger.error("Gemini가 빈 응답을 반환했습니다.")
                return "# AI 분석 보고서\n\n## 분석 오류\n\nGemini API가 응답을 생성하지 못했습니다."
            
    except Exception as e:
        logger.error(f"Gemini API 호출 중 오류 발생: {str(e)}")
        return f"# AI 분석 보고서\n\n## 분석 오류\n\nGemini API 호출 중 오류가 발생했습니다: {str(e)}"


def clean_markdown_format(text: str) -> str:
    """마크다운 형식을 정리하고 일관성을 높입니다."""
    lines = text.split('\n')
    result_lines = []
    
    # 불릿 포인트 형식 일관화 (* -> -)
    for i, line in enumerate(lines):
        # 불릿 포인트 일관화
        if line.strip().startswith('* '):
            line = line.replace('* ', '- ', 1)
        
        # 줄바꿈 개선: 제목 앞에는 빈 줄 추가
        if line.startswith('#') and i > 0 and lines[i-1].strip():
            result_lines.append('')
        
        # 현재 줄 추가
        result_lines.append(line)
        
        # 줄바꿈 개선: 제목 뒤에는 빈 줄 추가
        if line.startswith('#') and i < len(lines) - 1 and lines[i+1].strip() and not lines[i+1].startswith('#'):
            result_lines.append('')
    
    return '\n'.join(result_lines)