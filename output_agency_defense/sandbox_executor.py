"""
output_agency_defense/sandbox_executor.py
============================================
Docker-based sandbox for isolated tool execution.

Purpose:
    Execute untrusted tool calls in an isolated Docker container with:
    - No network access (network_mode=none)
    - Read-only filesystem
    - Memory and CPU limits
    - Execution timeout
    - Dropped privileges (no-new-privileges)

    This prevents a compromised tool from:
    - Exfiltrating data over the network
    - Modifying the host filesystem
    - Consuming excessive resources
    - Escalating privileges

Dependencies:
    - Docker must be installed and accessible
    - docker Python SDK: pip install docker

Usage:
    executor = SandboxExecutor()
    result = executor.execute("print('hello')")
    # result.stdout == "hello\n", result.exit_code == 0
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import docker
    from docker.errors import ContainerError, ImageNotFound, APIError
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_IMAGE = "python:3.11-slim"
DEFAULT_TIMEOUT = 10        # seconds
DEFAULT_MEM_LIMIT = "128m"
DEFAULT_CPU_QUOTA = 50000   # 50% of one CPU (period=100000)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass
class SandboxResult:
    """Result of sandboxed execution."""
    exit_code: int = -1
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    error: Optional[str] = None
    latency_ms: int = 0
    container_id: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and self.error is None


# ---------------------------------------------------------------------------
# Sandbox Executor
# ---------------------------------------------------------------------------
class SandboxExecutor:
    """
    Executes code/commands in an isolated Docker container.

    Security constraints:
    - network_mode="none"          → no network access
    - read_only=True               → immutable filesystem
    - mem_limit                    → memory cap
    - cpu_quota / cpu_period       → CPU cap
    - security_opt=no-new-privileges → no privilege escalation
    - auto_remove=True             → container cleaned up after execution
    """

    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        timeout: int = DEFAULT_TIMEOUT,
        mem_limit: str = DEFAULT_MEM_LIMIT,
        cpu_quota: int = DEFAULT_CPU_QUOTA,
    ):
        self.image = image
        self.timeout = timeout
        self.mem_limit = mem_limit
        self.cpu_quota = cpu_quota
        self._client: Optional[Any] = None

    def _get_client(self):
        """Lazy-initialize Docker client."""
        if not DOCKER_AVAILABLE:
            raise RuntimeError("docker SDK not installed. Run: pip install docker")
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    def is_available(self) -> bool:
        """Check if Docker is accessible."""
        if not DOCKER_AVAILABLE:
            return False
        try:
            client = self._get_client()
            client.ping()
            return True
        except Exception:
            return False

    def execute(
        self,
        code: str,
        language: str = "python",
    ) -> SandboxResult:
        """
        Execute code in a sandboxed Docker container.

        Args:
            code:     Code string to execute.
            language: Execution language ("python" or "bash").

        Returns:
            SandboxResult with stdout, stderr, exit_code.
        """
        t0 = time.time()

        if not DOCKER_AVAILABLE:
            return SandboxResult(
                error="docker SDK not installed",
                latency_ms=int((time.time() - t0) * 1000),
            )

        try:
            client = self._get_client()
        except Exception as e:
            return SandboxResult(
                error=f"Docker not accessible: {e}",
                latency_ms=int((time.time() - t0) * 1000),
            )

        # Build command
        if language == "python":
            cmd = ["python", "-c", code]
        elif language == "bash":
            cmd = ["bash", "-c", code]
        else:
            return SandboxResult(
                error=f"Unsupported language: {language}",
                latency_ms=int((time.time() - t0) * 1000),
            )

        try:
            container = client.containers.run(
                image=self.image,
                command=cmd,
                detach=True,
                network_mode="none",
                read_only=True,
                mem_limit=self.mem_limit,
                cpu_quota=self.cpu_quota,
                cpu_period=100000,
                security_opt=["no-new-privileges:true"],
                tmpfs={"/tmp": "size=32M"},
            )

            # Wait with timeout
            try:
                wait_result = container.wait(timeout=self.timeout)
                exit_code = wait_result.get("StatusCode", -1)
                timed_out = False
            except Exception:
                # Timeout — kill container
                try:
                    container.kill()
                except Exception:
                    pass
                exit_code = -1
                timed_out = True

            # Collect logs
            stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
            container_id = container.short_id

            # Cleanup
            try:
                container.remove(force=True)
            except Exception:
                pass

            latency_ms = int((time.time() - t0) * 1000)

            return SandboxResult(
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                timed_out=timed_out,
                latency_ms=latency_ms,
                container_id=container_id,
            )

        except ImageNotFound:
            return SandboxResult(
                error=f"Docker image not found: {self.image}. Run: docker pull {self.image}",
                latency_ms=int((time.time() - t0) * 1000),
            )
        except APIError as e:
            return SandboxResult(
                error=f"Docker API error: {e}",
                latency_ms=int((time.time() - t0) * 1000),
            )
        except Exception as e:
            return SandboxResult(
                error=f"Sandbox error: {e}",
                latency_ms=int((time.time() - t0) * 1000),
            )

    def execute_tool_code(
        self,
        tool_name: str,
        params: Dict[str, Any],
        code_template: str,
    ) -> SandboxResult:
        """
        Execute a tool call in the sandbox using a code template.

        Args:
            tool_name:     Name of the tool being executed.
            params:        Tool parameters (injected into template).
            code_template: Python code template with {params} placeholder.

        Returns:
            SandboxResult with execution output.
        """
        import json
        safe_params = json.dumps(params)
        code = code_template.replace("{params}", safe_params)
        code = code.replace("{tool_name}", tool_name)
        return self.execute(code)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    executor = SandboxExecutor()

    print(f"{'='*60}")
    print(f"  SANDBOX EXECUTOR DEMO")
    print(f"  Docker available: {executor.is_available()}")
    print(f"  Image: {executor.image}")
    print(f"  Timeout: {executor.timeout}s | Mem: {executor.mem_limit}")
    print(f"{'='*60}")

    if not executor.is_available():
        print("\n  Docker is not available. Skipping execution tests.")
        print("  Make sure Docker is installed and running.")
        exit(0)

    # Test 1: Simple code
    print(f"\n  [Test 1] Simple Python code:")
    result = executor.execute("print('Hello from sandbox!')")
    print(f"    Exit: {result.exit_code} | Stdout: {result.stdout.strip()}")
    print(f"    Latency: {result.latency_ms}ms")

    # Test 2: Math computation
    print(f"\n  [Test 2] Computation:")
    result = executor.execute("import math; print(f'pi = {math.pi:.6f}')")
    print(f"    Exit: {result.exit_code} | Stdout: {result.stdout.strip()}")

    # Test 3: Network blocked
    print(f"\n  [Test 3] Network access (should fail):")
    result = executor.execute(
        "import urllib.request; urllib.request.urlopen('http://example.com')"
    )
    print(f"    Exit: {result.exit_code} | Has error: {bool(result.stderr)}")
    if result.stderr:
        print(f"    Stderr: {result.stderr[:80]}...")

    # Test 4: Filesystem read-only
    print(f"\n  [Test 4] Write to filesystem (should fail):")
    result = executor.execute("open('/data.txt', 'w').write('hack')")
    print(f"    Exit: {result.exit_code} | Has error: {bool(result.stderr)}")

    # Test 5: Write to /tmp (allowed via tmpfs)
    print(f"\n  [Test 5] Write to /tmp (tmpfs, should succeed):")
    result = executor.execute(
        "open('/tmp/test.txt','w').write('ok'); print(open('/tmp/test.txt').read())"
    )
    print(f"    Exit: {result.exit_code} | Stdout: {result.stdout.strip()}")

    # Test 6: Timeout
    print(f"\n  [Test 6] Timeout (infinite loop):")
    executor_short = SandboxExecutor(timeout=3)
    result = executor_short.execute("import time; time.sleep(30)")
    print(f"    Timed out: {result.timed_out} | Latency: {result.latency_ms}ms")

    # Test 7: Tool code template
    print(f"\n  [Test 7] Tool code template:")
    result = executor.execute_tool_code(
        tool_name="get_order",
        params={"resource_id": "ORD-001"},
        code_template="""
import json
params = json.loads('{params}')
print(f"Tool: {tool_name}")
print(f"Params: {{params}}")
print(f"Result: Order {{params['resource_id']}} found")
""",
    )
    print(f"    Exit: {result.exit_code}")
    for line in result.stdout.strip().split("\n"):
        print(f"    {line}")

    print(f"\n{'='*60}")
