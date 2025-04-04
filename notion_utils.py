import os
import json
import logging
import httpx
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
from dotenv import load_dotenv
from notion_markdown import create_markdown_blocks

# 환경 변수 로드
load_dotenv()

# Notion API 설정
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
REFERENCE_DB_ID = os.getenv("REFERENCE_DB_ID")
SCRIPT_DB_ID = os.getenv("SCRIPT_DB_ID")

logger = logging.getLogger(__name__)

async def query_notion_database(database_id: str, request_body: dict = None, max_retries: int = 3, timeout: float = 30.0) -> List[Dict[str, Any]]:
    """
    Notion 데이터베이스를 쿼리합니다. 재시도 및 타임아웃 처리가 포함되어 있습니다.
    """
    url = f"https://api.notion.com/v1/databases/{database_id}/query"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }
    
    if request_body is None:
        request_body = {}
    
    # 재시도 로직
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                logger.info(f"Querying Notion database (attempt {attempt+1}/{max_retries})")
                response = await client.post(
                    url, 
                    headers=headers, 
                    json=request_body, 
                    timeout=timeout
                )
                
                response.raise_for_status()
                results = response.json().get("results", [])
                logger.info(f"Successfully retrieved {len(results)} records from Notion database")
                return results
                
        except httpx.TimeoutException:
            logger.warning(f"Timeout when querying Notion database (attempt {attempt+1}/{max_retries})")
            if attempt < max_retries - 1:
                # 재시도 전 잠시 대기 (지수 백오프)
                await asyncio.sleep(2 ** attempt)
            else:
                logger.error(f"Max retries reached when querying Notion database {database_id}")
                return []
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            # 429 (Rate Limit) 오류인 경우 더 오래 대기
            if e.response.status_code == 429 and attempt < max_retries - 1:
                retry_after = int(e.response.headers.get("Retry-After", 5))
                logger.warning(f"Rate limited. Waiting for {retry_after}s before retry")
                await asyncio.sleep(retry_after)
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error querying Notion database: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                return []
    
    return []


async def create_script_report_page(database_id: str, properties: Dict[str, Any], content: str, max_retries: int = 3, timeout: float = 120.0) -> Optional[Dict[str, Any]]:
    """
    Notion에 새 페이지를 생성합니다. 마크다운 형식의 콘텐츠를 적절한 Notion 블록으로 변환합니다.
    재시도 및 타임아웃 처리가 포함되어 있습니다.
    """
    url = "https://api.notion.com/v1/pages"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }
    
    # 페이지 내용 설정 - 개선된 마크다운 처리 사용
    data = {
        "parent": {"database_id": database_id},
        "properties": properties,
        "children": create_markdown_blocks(content)
    }
    
    # Notion API는 한 번에 100개의 블록까지만 허용
    # 블록이 100개 이상이면 나눠서 요청
    MAX_BLOCKS_PER_REQUEST = 90  # 안전하게 90개로 제한
    
    if len(data["children"]) > MAX_BLOCKS_PER_REQUEST:
        logger.info(f"블록이 너무 많아 여러 요청으로 나누어 처리합니다. 총 {len(data['children'])}개 블록")
        
        # 첫 번째 요청: 속성과 첫 90개 블록
        first_request_data = {
            "parent": data["parent"],
            "properties": data["properties"],
            "children": data["children"][:MAX_BLOCKS_PER_REQUEST]
        }
        
        # 첫 번째 페이지 생성
        page_response = None
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient() as client:
                    logger.info(f"Creating Notion page - first part (attempt {attempt+1}/{max_retries})")
                    response = await client.post(
                        url, 
                        headers=headers, 
                        json=first_request_data,
                        timeout=timeout
                    )
                    
                    response.raise_for_status()
                    page_response = response.json()
                    logger.info(f"First part created successfully")
                    break
                    
            except Exception as e:
                logger.error(f"Error creating first part: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return None
        
        if not page_response:
            return None
            
        # 남은 블록을 90개씩 나눠 추가 요청
        page_id = page_response["id"]
        remaining_blocks = data["children"][MAX_BLOCKS_PER_REQUEST:]
        
        for i in range(0, len(remaining_blocks), MAX_BLOCKS_PER_REQUEST):
            append_blocks = remaining_blocks[i:i + MAX_BLOCKS_PER_REQUEST]
            
            # 블록 추가 요청
            append_url = f"https://api.notion.com/v1/blocks/{page_id}/children"
            append_data = {"children": append_blocks}
            
            success = False
            for attempt in range(max_retries):
                try:
                    async with httpx.AsyncClient() as client:
                        logger.info(f"Appending blocks part {i//MAX_BLOCKS_PER_REQUEST + 2} (attempt {attempt+1}/{max_retries})")
                        response = await client.patch(
                            append_url, 
                            headers=headers, 
                            json=append_data,
                            timeout=timeout
                        )
                        
                        response.raise_for_status()
                        logger.info(f"Part {i//MAX_BLOCKS_PER_REQUEST + 2} appended successfully")
                        success = True
                        # API 제한 준수를 위한 딜레이
                        await asyncio.sleep(0.5)  # 0.5초 대기
                        break
                        
                except Exception as e:
                    logger.error(f"Error appending part {i//MAX_BLOCKS_PER_REQUEST + 2}: {str(e)}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        # 실패해도 계속 진행, 일부 콘텐츠라도 저장
                        logger.warning(f"Failed to append part {i//MAX_BLOCKS_PER_REQUEST + 2}, but continuing")
            
            if not success:
                logger.warning(f"Could not append all blocks to page")
                
        return page_response
    
    # 블록이 적은 경우 단일 요청으로 처리
    else:
        # 재시도 로직
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient() as client:
                    logger.info(f"Creating Notion page (attempt {attempt+1}/{max_retries})")
                    response = await client.post(
                        url, 
                        headers=headers, 
                        json=data,
                        timeout=timeout
                    )
                    
                    # 디버깅을 위한 상세 오류 로깅
                    if response.status_code != 200:
                        logger.error(f"Notion API 오류: {response.status_code}")
                        logger.error(f"응답 내용: {response.text}")
                        
                        # 오류 내용 상세 분석
                        try:
                            error_json = response.json()
                            if "message" in error_json:
                                logger.error(f"API 오류 메시지: {error_json['message']}")
                            if "code" in error_json:
                                logger.error(f"API 오류 코드: {error_json['code']}")
                        except:
                            pass
                    
                    response.raise_for_status()
                    logger.info(f"Successfully created Notion page")
                    return response.json()
                    
            except httpx.TimeoutException:
                logger.warning(f"Timeout when creating Notion page (attempt {attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error("Max retries reached when creating Notion page")
                    return None
                    
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
                # 429 (Rate Limit) 오류인 경우 더 오래 대기
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    retry_after = int(e.response.headers.get("Retry-After", 5))
                    logger.warning(f"Rate limited. Waiting for {retry_after}s before retry")
                    await asyncio.sleep(retry_after)
                else:
                    return None
                    
            except Exception as e:
                logger.error(f"Error creating Notion page: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error("Max retries reached")
                    return None
        
        return None
        
async def update_notion_page(page_id: str, properties: Dict[str, Any], max_retries: int = 3, timeout: float = 30.0) -> bool:
    """
    Notion 페이지의 속성을 업데이트합니다.
    재시도 및 타임아웃 처리가 포함되어 있습니다.
    """
    url = f"https://api.notion.com/v1/pages/{page_id}"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }
    
    data = {
        "properties": properties
    }
    
    # 재시도 로직
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                logger.info(f"Updating Notion page (attempt {attempt+1}/{max_retries})")
                response = await client.patch(
                    url, 
                    headers=headers, 
                    json=data,
                    timeout=timeout
                )
                
                response.raise_for_status()
                logger.info(f"Successfully updated Notion page")
                return True
                
        except httpx.TimeoutException:
            logger.warning(f"Timeout when updating Notion page (attempt {attempt+1}/{max_retries})")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                logger.error("Max retries reached when updating Notion page")
                return False
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            if e.response.status_code == 429 and attempt < max_retries - 1:
                retry_after = int(e.response.headers.get("Retry-After", 5))
                await asyncio.sleep(retry_after)
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error updating Notion page: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                return False
    
    return False

async def check_script_exists(video_url: str) -> bool:
    """스크립트 DB에 이미 해당 영상의 스크립트가 있는지 확인합니다."""
    script_pages = await query_notion_database(SCRIPT_DB_ID)
    
    for page in script_pages:
        properties = page.get("properties", {})
        url_property = properties.get("URL", {})
        
        if "url" in url_property and url_property["url"] == video_url:
            return True
    
    return False

async def reset_all_channels() -> bool:
    """참고용 DB의 모든 채널을 활성화 상태로 변경합니다."""
    reference_pages = await query_notion_database(REFERENCE_DB_ID)
    logger.info(f"Resetting {len(reference_pages)} channels to active state")
    
    success_count = 0
    
    for page in reference_pages:
        page_id = page.get("id")
        properties = {
            "활성화": {
                "checkbox": True
            }
        }
        
        success = await update_notion_page(page_id, properties)
        if success:
            success_count += 1
    
    logger.info(f"Successfully reset {success_count}/{len(reference_pages)} channels")
    return success_count > 0